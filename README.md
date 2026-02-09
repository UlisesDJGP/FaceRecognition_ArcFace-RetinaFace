# Face Recognition using ArcFace and RetinaFace

## Overview
This project implements face recognition leveraging the ArcFace and RetinaFace algorithms. ArcFace is known for its accuracy and robustness in distinguishing faces, while RetinaFace serves as an efficient face detection model.

## Features
- High accuracy face recognition using ArcFace.
- Quick and efficient face detection using RetinaFace.
- Easy to set up and use.

## Architecture
The project consists of:
1. **Face Detection**: Implemented using RetinaFace, which extracts faces from images.
2. **Face Recognition**: Utilizes the ArcFace algorithm to recognize and verify faces.
3. **Integration**: Both components work together seamlessly to deliver an accurate face recognition solution.

## Installation
To install the necessary components, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/HilkirySG/FaceRecognition_ArcFace-RetinaFace.git
   ```
2. Navigate to the project directory:
   ```bash
   cd FaceRecognition_ArcFace-RetinaFace
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the face recognition feature, run the following command:
```bash
python main.py --image path/to/image.jpg
```
Replace `path/to/image.jpg` with the path to the image you want to process.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.