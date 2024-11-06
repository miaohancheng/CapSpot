# CapSpot
CAPTCHA Resolution Tool

[![PyPI version](https://badge.fury.io/py/CapSpot.svg?icon=si%3Apython&icon_color=%23ffffff)](https://badge.fury.io/py/CapSpot)
[![License](https://img.shields.io/github/license/miaohancheng/CapSpot)](https://github.com/miaohancheng/CapSpot/blob/main/LICENSE)

## Introduction

`CapSpot` is an image matching library based on ONNX models that locates and identifies sub-images within a target image using a template. It is designed to solve CAPTCHA challenges by finding specific click points in images, making it an invaluable tool for automated testing and security applications.

## Example Outputs
### Example 1
- **Target Image**

![test.png](test_pic/test.png)

- **Detection Template and Order**

![test_mo.png](test_pic/test_mo.png)

- **Detection Results**

![result.png](test_pic/result.png)

- **JSON Response**
```json
{'img0': [103, 35, 145, 76], 'img1': [162, 90, 196, 127], 'img2': [175, 189, 199, 218]}
```

### Example 2
- **Target Image**

  ![test1.png](test_pic/test1.png)
- **Detection Template and Order**

  ![test1_mo.png](test_pic/test1_mo.png)
- **Detection Results**

![result1.png](test_pic/result1.png)

- **JSON Response**
```json
{'img0': [24, 28, 89, 93], 'img1': [243, 60, 300, 119], 'img2': [134, 128, 191, 184]}
```


## Features

- - Utilizes pre-trained ONNX models for object detection and similarity calculations.
- - Supports optional GPU acceleration.
- - Offers a simple and easy-to-use API.

## Installation
### Via pip:
```bash 
pip install CapSpot
```

### From source:
```bash
git clone https://github.com/miaohancheng/CapSpot.git
cd CapSpot
python setup.py install
```

## Usage
```python
from CapSpot import ImageMatcher

if __name__ == '__main__':
    matcher = ImageMatcher()
    matches = matcher.match_images('./test_pic/test1_mo.png', './test_pic/test1.png')
    matcher.display_results('./test_pic/test1.png', matches)
    print(matches)

```

### Configuration Parameters

	•	ImageMatcher class
	•	sim_model_name: Name of the similarity model (ONNX format), with a default model provided.
	•	det_model_name: Name of the detection model (ONNX format), with a default model provided.
	•	match_images method
	•	template_file: Path to the template image.
	•	target_file: Path to the target image.

## Model Downloads

Place the required model files (similarity_model.onnx and detection_model.onnx) in the models/ directory.

## Dependencies

	•	Python 3.6 or higher
	•	numpy
	•	opencv-python
	•	Pillow
	•	onnxruntime

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

	•	Author: Miao HanCheng
	•	Email: hanchengmiao@gmail.com