 Domain
Artificial Intelligence / Deep Learning

🔧 Technologies Used
Python
TensorFlow & Keras
Convolutional Neural Networks (CNN)
Transfer Learning (MobileNetV2)
Google Colab
🎯 Objective
To build an AI-powered rice type classification model that can accurately identify 5 types of rice grains (Arborio, Basmati, Ipsala, Jasmine, Karacadag) using deep learning and transfer learning techniques. This solution aids farmers, scientists, and consumers in rice identification for agricultural decision-making.

📂 Dataset
Name: Rice Image Dataset
Images: 75,000+ across 5 classes
Preprocessing: Resizing (224x224), normalization, augmentation
🧠 Model Architecture
Base: MobileNetV2 (pre-trained on ImageNet)
Custom Head: GlobalAveragePooling + Dense(Softmax)
Optimizer: RMSProp
Loss Function: Categorical Crossentropy
Validation Accuracy: ~96.4%
📈 Evaluation Metrics
Accuracy, Loss (visualized with matplotlib)
Consistently >94% validation accuracy
🧪 Testing
Real-time prediction from uploaded rice grain image
Results displayed directly in Colab notebook
💾 Output Files
rice_classifier_model.keras — Saved trained model
prediction_notebook.ipynb — Colab notebook to predict from image
🚀 Deployment Ready
Can be deployed using:

Streamlit or Gradio Web App
TensorFlow Lite for Android
Flask API for web integrations
📁 Suggested Repository Structure
Rice-Type-Classifier/
├── README.md
├── rice_classifier_model.keras
├── prediction_notebook.ipynb
├── training_notebook.ipynb
├── rice_dataset/ (optional)
├── utils/ (optional)
└── requirements.txt
👥 Team
Group 

📝 Notes
This project showcases the power of transfer learning in agriculture. In just 3 epochs, a highly accurate model was achieved — making it practical for real-world deployment.
