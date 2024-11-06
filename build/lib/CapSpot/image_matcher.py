import io
import numbers
import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

class ImageMatcher:
    def __init__(self, sim_model_name='sim_model.onnx', det_model_name='det_model.onnx'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sim_model_path = os.path.join(current_dir, 'models', sim_model_name)
        det_model_path = os.path.join(current_dir, 'models', det_model_name)

        self.onnx_model = ONNXModel(sim_model_path)
        self.det_model = DetectModel(det_model_path)

    def preprocess_input(self, x):
        return x / 255.0

    def decouple(self, image):
        pixel_values = image.flatten()
        counts = np.bincount(pixel_values)
        max_pixel = np.argmax(counts) - 5
        mask = (image > max_pixel + 10) | (image < max_pixel - 10)
        image[mask] = 0
        _, binary = cv2.threshold(image, max_pixel, 255, cv2.THRESH_BINARY)
        return binary

    def extract_and_match(self, image_data):
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        bboxes = self.det_model.get_bbox(image_data)
        bboxes_sorted = sorted(bboxes, key=lambda bbox: bbox[0])
        img_results = []
        for (x, y, x2, y2) in bboxes_sorted:
            cropped_img = img[y:y2, x:x2]
            _, encoded_img = cv2.imencode('.png', cropped_img)
            img_results.append(encoded_img)
        return img_results, bboxes_sorted

    def match_images(self, template_image_data, target_image_data):
        template_list, _ = self.extract_and_match(template_image_data)
        target_list, target_boxes = self.extract_and_match(target_image_data)
        resp = {}
        used_targets = set()  # 用来记录已匹配的目标图像索引
        for i, template_encoded_image in enumerate(template_list):
            max_probability = 0
            best_match = None
            template_image = Image.open(io.BytesIO(template_encoded_image.tobytes()))
            for j, target_encoded_image in enumerate(target_list):
                if j in used_targets:  # 检查索引是否已经被使用
                    continue
                target_image = Image.open(io.BytesIO(target_encoded_image.tobytes()))
                probability = self.onnx_model.detect_image(template_image, target_image)
                if probability > max_probability:
                    max_probability = probability
                    best_match = j
            if best_match is not None:
                resp[f'img{i}'] = target_boxes[best_match]
                used_targets.add(best_match)  # 添加索引到已使用集合
        return resp

    def display_results(self, background_image_data, match_results):
        background_img = cv2.imdecode(np.frombuffer(background_image_data, np.uint8), 1)
        for idx, coords in match_results.items():
            cv2.rectangle(background_img, tuple(coords[:2]), tuple(coords[2:]), (0, 0, 255), 2)
            cv2.putText(background_img, idx, ((coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
        _, result_encoded = cv2.imencode('.png', background_img)
        return io.BytesIO(result_encoded.tobytes())

class ONNXModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    def preprocess_input(self, x):
        return x / 255.0
    def detect_image(self, image_1, image_2):
        processed_image_1 = self.process_image(image_1)
        processed_image_2 = self.process_image(image_2)
        input_feed = {self.input_name: processed_image_1, self.session.get_inputs()[1].name: processed_image_2}
        output = np.array(self.session.run([self.output_name], input_feed)[0])
        return output

    def process_image(self, image):
        image = self.cvtColor(image)
        image = self.letterbox_image(image, [60, 60], True)
        photo = np.expand_dims(self.preprocess_input(np.array(image, dtype=np.float32)), axis=0)
        return photo

    def letterbox_image(self, image, size, convert):
        w, h = size
        iw, ih = image.size
        if convert:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            if h == w:
                new_image = self.resize(image, h)
            else:
                new_image = self.resize(image, [h, w])
            new_image = self.center_crop(new_image, [h, w])
        return new_image

    def cvtColor(self, image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    def crop(self, img, i, j, h, w):
        return img.crop((j, i, j + w, i + h))

    def center_crop(self, img, output_size):
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        w, h = img.size
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return self.crop(img, i, j, th, tw)

    def resize(self, img, size, interpolation=Image.BILINEAR):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

class DetectModel:
    def __init__(self, model_path, use_gpu=False):
        providers = [('CUDAExecutionProvider', {'device_id': 0})] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_image = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_image = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_image = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_image[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_image

        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
        return padded_image, r

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        return self.multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr)

    def get_bbox(self, image_bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        im, ratio = self.preproc(img, (416, 416))
        ort_inputs = {self.session.get_inputs()[0].name: im[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        predictions = self.demo_postprocess(output[0], (416, 416))[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        pred = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if pred is None:
            return []
        final_boxes = pred[:, :4].tolist()
        result = []
        img_shape = img.shape
        for b in final_boxes:
            x_min = max(0, int(b[0]))
            y_min = max(0, int(b[1]))
            x_max = min(img_shape[1], int(b[2]))
            y_max = min(img_shape[0], int(b[3]))
            result.append([x_min, y_min, x_max, y_max])
        return result

if __name__ == '__main__':
    matcher = ImageMatcher()
    with open('../test_pic/test1_mo.png', 'rb') as template_file, open('../test_pic/test1.png', 'rb') as target_file:
        template_image_data = template_file.read()
        target_image_data = target_file.read()
    matches = matcher.match_images(template_image_data, target_image_data)
    result_stream = matcher.display_results(target_image_data, matches)
    with open('result.png', 'wb') as f:
        f.write(result_stream.read())
    print(matches)