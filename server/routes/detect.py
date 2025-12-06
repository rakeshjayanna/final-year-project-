from flask import Blueprint, request, jsonify
from PIL import Image
import numpy as np
from pathlib import Path
import json
import joblib

detect_bp = Blueprint('detect', __name__)

# Multi-task model cache: {task_name: {model, feature_extractor, svm_model, input_size, class_names}}
MODEL_CACHE = {}


def _model_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'model'


def _artifacts_dir(task: str) -> Path:
    return _model_dir() / 'artifacts' / task


def _load_labels(artifacts_dir: Path) -> list[str] | None:
    labels_file = artifacts_dir / 'class_indices.json'
    if labels_file.exists():
        try:
            mapping = json.loads(labels_file.read_text())
            items = sorted(((int(k), v) for k, v in mapping.items()), key=lambda kv: kv[0])
            return [name for _, name in items]
        except Exception:
            pass
    return None


def _comparison_json_path(task: str) -> Path:
    return _artifacts_dir(task) / 'metrics' / 'model_comparison.json'


def _best_model_json_path(task: str) -> Path:
    return _artifacts_dir(task) / 'best_model.json'


def _get_best_model_name(task: str) -> str | None:
    p = _best_model_json_path(task)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data.get('best_model')
        except Exception:
            return None
    return None


def _load_task_models(task: str):
    """Load models for a specific task into the cache"""
    if task in MODEL_CACHE:
        return  # Already loaded
    
    import tensorflow as tf
    artifacts_dir = _artifacts_dir(task)
    model_path = artifacts_dir / 'mango_model.h5'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first with: python server/model/model_trainer.py --task {task}")
    
    # Load CNN
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    # Determine input size
    try:
        ishape = model.inputs[0].shape
        h = int(ishape[1]) if ishape[1] is not None else 224
        w = int(ishape[2]) if ishape[2] is not None else 224
        input_size = (h, w)
    except Exception:
        input_size = (224, 224)
    
    # Load class names
    class_names = _load_labels(artifacts_dir)
    
    # Build feature extractor
    feature_extractor = None
    try:
        penultimate = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                penultimate = layer
                break
        output_tensor = penultimate.output if penultimate is not None else model.layers[-2].output
        feature_extractor = tf.keras.Model(inputs=model.input, outputs=output_tensor)
    except Exception:
        feature_extractor = None
    
    # Load SVM
    svm_model = None
    svm_path = artifacts_dir / 'models' / 'svm.pkl'
    if svm_path.exists():
        try:
            svm_model = joblib.load(svm_path)
        except Exception:
            svm_model = None
    
    # Cache all models for this task
    MODEL_CACHE[task] = {
        'model': model,
        'feature_extractor': feature_extractor,
        'svm_model': svm_model,
        'input_size': input_size,
        'class_names': class_names,
    }


def _get_task_models(task: str):
    """Get cached models for a task, loading if necessary"""
    if task not in MODEL_CACHE:
        _load_task_models(task)
    return MODEL_CACHE[task]


def _reset_models():
    """Clear the entire model cache"""
    global MODEL_CACHE
    MODEL_CACHE = {}


def _preprocess_image(file_storage, input_size) -> np.ndarray:
    img = Image.open(file_storage.stream).convert('RGB')
    img = img.resize((input_size[1], input_size[0]))
    x = np.array(img, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def _preprocess_raw_image(file_storage, input_size) -> np.ndarray:
    img = Image.open(file_storage.stream).convert('RGB')
    img = img.resize((input_size[1], input_size[0]))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def _validate_mango_image(cnn_conf, svm_conf, threshold=0.50):
    confidences = [c for c in [cnn_conf, svm_conf] if c is not None]
    if not confidences:
        return False, "No valid predictions available"
    
    max_conf = max(confidences)
    avg_conf = sum(confidences) / len(confidences)
    
    if max_conf < threshold:
        return False, f"Low confidence ({max_conf*100:.1f}%) - image may not be a mango"
    
    if avg_conf < (threshold * 0.8):  # 40% for 50% threshold
        return False, f"Average confidence too low ({avg_conf*100:.1f}%) - image may not be a mango"
    
    return True, "Valid mango image"


def _compare_on_image(file_storage, task: str):
    # Load task-specific models
    task_models = _get_task_models(task)
    MODEL = task_models['model']
    FEATURE_EXTRACTOR = task_models['feature_extractor']
    SVM_MODEL = task_models['svm_model']
    INPUT_SIZE = task_models['input_size']
    CLASS_NAMES = task_models['class_names']

    x_scaled = _preprocess_image(file_storage, INPUT_SIZE)
    preds = MODEL.predict(x_scaled, verbose=0)
    if preds.shape[-1] == 1:
        p1 = float(preds[0][0])
        cnn_idx = 1 if p1 >= 0.5 else 0
        cnn_conf = p1 if cnn_idx == 1 else (1.0 - p1)
        cnn_probs = None
        if CLASS_NAMES and len(CLASS_NAMES) == 2:
            try:
                cnn_probs = {
                    CLASS_NAMES[0]: round((1.0 - p1) * 100.0, 2),
                    CLASS_NAMES[1]: round(p1 * 100.0, 2),
                }
            except Exception:
                cnn_probs = None
    else:
        import tensorflow as tf
        probs = tf.nn.softmax(preds[0]).numpy().tolist()
        cnn_idx = int(np.argmax(probs))
        cnn_conf = float(probs[cnn_idx])
        try:
            if CLASS_NAMES and len(CLASS_NAMES) == len(probs):
                cnn_probs = {CLASS_NAMES[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(probs))}
            else:
                cnn_probs = {str(i): round(float(probs[i]) * 100.0, 2) for i in range(len(probs))}
        except Exception:
            cnn_probs = None
    cnn_label = CLASS_NAMES[cnn_idx] if CLASS_NAMES and cnn_idx < len(CLASS_NAMES) else str(cnn_idx)

    xr = _preprocess_raw_image(file_storage, INPUT_SIZE)
    feats = FEATURE_EXTRACTOR.predict(xr, verbose=0) if FEATURE_EXTRACTOR is not None else None
    svm_label = svm_conf = None
    svm_probs = None
    if feats is not None and SVM_MODEL is not None:
        try:
            if hasattr(SVM_MODEL, 'predict_proba'):
                proba = SVM_MODEL.predict_proba(feats)[0]
                si = int(np.argmax(proba))
                svm_conf = float(proba[si])
                try:
                    if CLASS_NAMES and len(CLASS_NAMES) == len(proba):
                        svm_probs = {CLASS_NAMES[i]: round(float(proba[i]) * 100.0, 2) for i in range(len(proba))}
                    else:
                        svm_probs = {str(i): round(float(proba[i]) * 100.0, 2) for i in range(len(proba))}
                except Exception:
                    svm_probs = None
            else:
                si = int(SVM_MODEL.predict(feats)[0])
                svm_conf = 1.0
            svm_label = CLASS_NAMES[si] if CLASS_NAMES and si < len(CLASS_NAMES) else str(si)
        except Exception:
            svm_label = svm_conf = None

    is_valid, validation_msg = _validate_mango_image(cnn_conf, svm_conf)
    if not is_valid:
        return {'error': validation_msg, 'is_mango': False}

    models = {
        'cnn': {'label': cnn_label, 'confidence': round(cnn_conf * 100.0, 2), 'probs': cnn_probs},
    }
    if svm_label is not None and svm_conf is not None:
        models['svm'] = {'label': svm_label, 'confidence': round(svm_conf * 100.0, 2), 'probs': svm_probs}

    reason = ''
    best_model = 'cnn'
    accs = {'cnn': None, 'svm': None}
    comp_path = _comparison_json_path(task)
    if comp_path.exists():
        try:
            comp = json.loads(comp_path.read_text())
            accs['cnn'] = float(comp['models'].get('cnn', {}).get('accuracy', 0.0))
            accs['svm'] = float(comp['models'].get('svm', {}).get('accuracy', 0.0))
            available = {k: v for k, v in accs.items() if k in models and v is not None}
            if available:
                best_model = max(available.keys(), key=lambda k: available[k])
                reason = 'highest validation accuracy'
        except Exception:
            pass
    if not reason:
        best_model = max(models.keys(), key=lambda k: models[k]['confidence'])
        reason = 'highest confidence on this image'

    final = models.get(best_model)
    selection = {
        'model': best_model,
        'reason': reason,
        'detail': {
            'cnn_acc': accs['cnn'] if accs['cnn'] is not None else 0.0,
            'svm_acc': accs['svm'] if accs['svm'] is not None else 0.0,
        }
    }

    return {'models': models, 'selection': selection, 'final': final}


@detect_bp.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image provided'}), 400
        
        # Get task from form data
        task = request.form.get('task', 'disease')  # Default to 'disease' for backward compatibility
        
        res = _compare_on_image(request.files['image'], task)
        
        if 'error' in res and 'is_mango' in res:
            return jsonify(res), 400
        
        final = res.get('final') or {}
        payload = {
            'label': final.get('label'),
            'confidence': final.get('confidence'),
            'model_used': res.get('selection', {}).get('model'),
            'task': task
        }
        payload['models'] = res.get('models')
        payload['selection'] = res.get('selection')
        return jsonify(payload)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'prediction failed: {str(e)}'}), 500


@detect_bp.route('/models/comparison', methods=['GET'])
def models_comparison():
    task = request.args.get('task', 'disease')  # Default to 'disease'
    p = _comparison_json_path(task)
    if not p.exists():
        return jsonify({'error': f'comparison metrics not found for task "{task}". Run: python server/model/compare_models.py --task {task}'}), 404
    try:
        data = json.loads(p.read_text())
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'failed to read comparison metrics: {str(e)}'}), 500


@detect_bp.route('/reload', methods=['POST'])
def reload_models():
    try:
        _reset_models()
        task = request.json.get('task') if request.json else None
        
        if task:
            # Reload specific task
            _load_task_models(task)
            return jsonify({
                'status': 'reloaded',
                'task': task,
                'model_present': True,
            })
        else:
            # Just clear cache, models will be loaded on demand
            return jsonify({
                'status': 'cache_cleared',
                'message': 'Model cache cleared. Models will be loaded on next request.',
            })
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@detect_bp.route('/compare-image', methods=['POST'])
def compare_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image provided'}), 400
        
        # Get task from form data
        task = request.form.get('task', 'disease')
        
        res = _compare_on_image(request.files['image'], task)
        
        if 'error' in res and 'is_mango' in res:
            return jsonify(res), 400
        
        res['task'] = task
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': f'compare failed: {str(e)}'}), 500


@detect_bp.route('/health', methods=['GET'])
def health():
    task = request.args.get('task', 'disease')
    artifacts_dir = _artifacts_dir(task)
    model_path = artifacts_dir / 'mango_model.h5'
    best = _get_best_model_name(task)
    
    return jsonify({
        'status': 'ok',
        'task': task,
        'model_present': model_path.exists(),
        'best_model': best,
    })
