# Cyberbullying and Harassment Classification
This project implements a robust and explainable NLP model for fine-grained multiclass classification of cyberbullying and harassment in online text. It identifies various types of toxic content including ethnic, religious, sexual, political, and vocational abuse, among others.

# Overview
With the rapid rise of online communication, detecting and moderating harmful content has become critical. This project leverages transformer-based models like RoBERTa (with LSTM hybridization) to classify tweets/posts into multiple harassment categories.

# Why This Project?
While mainstream NLP research has advanced significantly for languages like English, Hindi, and Spanish, lesser-known languages such as Tulu and Konkani remain underrepresented. These languages face a lack of labeled datasets and linguistic tools, making it difficult to build effective AI moderation solutions for speakers of these languages.

To simulate this low-resource environment, we conduct our study in the English language but intentionally limit our dataset to 8,000 samples, with approximately 500 samples per class. This reflects the real-world challenge of developing AI systems where data availability is limited.

Despite the limited dataset, our RoBERTa + LSTM hybrid model achieves strong performance with minimal overfitting, demonstrating its capability to generalize well even in data-scarce scenarios. This shows promise for extending similar techniques to truly low-resource languages in future work.

# Dataset
The dataset used contains 8,000 labeled samples with the following columns:

Text: The tweet or comment content.

Label: Binary label (0: Non-harassing, 1: Harassing).

Type: Harassment type (e.g., Ethnicity, Sexual, Religion, Troll, Threat, Vocational, Political).

# Results
Accuracy	        94.63%
Precision	        94.98%
Recall	          94.63%
F1-Score	        94.68%

# Model Architecture
# We use a hybrid NLP architecture:

RoBERTa for contextualized text embeddings.

LSTM layer for capturing sequential patterns.

Dense output layer with softmax for multiclass classification.

# Features:
Fine-grained multiclass classification.

HuggingFace Transformers & Trainer API.

Precision, Recall, F1-Score, and Confusion Matrix evaluation.

# Future Work
To further enhance the effectiveness and real-world applicability of our cyberbullying and harassment detection system, we propose the following directions for future work:
# Network Simulation with Mininet & POX
We plan to simulate real-time detection and mitigation of cyberbullying in a Software-Defined Networking (SDN) environment using tools like Mininet and POX controller. This would allow us to test how our classifier can be deployed in a network to take automated moderation actions.

🔧 How It Works:
Mininet is used to emulate a realistic network with multiple users, simulating chat applications or social media traffic.
POX, a Python-based SDN controller, will act as a decision-making agent that receives messages or posts from users.
The trained cyberbullying classifier will be integrated with the POX controller to:
Inspect messages in real time.
Classify them as offensive or non-offensive.
Take action such as blocking, flagging, or redirecting harmful content.
This approach can simulate how a smart content filter can operate at the network level, enabling early detection and control of online abuse in an infrastructure-  like setting—without compromising user privacy.









