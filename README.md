# Multilingual Seq2Seq Chatbot (English & Vietnamese)

This is a chatbot project using a Sequence-to-Sequence (Seq2Seq) model built with PyTorch. The chatbot is designed to support both English and Vietnamese through a simple interface.

## Key Features

- **Multilingual Support**: Automatic detection and switching between English and Vietnamese
- **Seq2Seq Architecture**: Using an encoder-decoder model with GRU
- **Unified Interface**: A single file for the chat interface (`chat/unified_chat.py`)
- **Pre-trained Models**: Available models for both English and Vietnamese

## Project Structure

```
.
├── chat/                      # Chat interface
│   └── unified_chat.py        # Unified chat interface with multilingual support
├── data/                      # Training data
│   ├── conversation.txt       # Original English data
│   ├── conversation_new.txt   # Extended English data
│   └── conversation_vietnamese.txt  # Vietnamese data
├── models/                    # Trained models
│   ├── chatbot.pt             # Initial English model
│   ├── chatbot_new.pt         # Improved English model
│   ├── vietnamese_chatbot.pt  # Vietnamese model
│   └── ...                    # Dictionaries and model checkpoints
├── src/                       # Source code
│   ├── model.py               # Seq2Seq model architecture definition
│   └── preprocess.py          # Input data processing
├── tools/                     # Tools and utilities
│   ├── check_model.py         # Model checker
│   ├── download_nltk_resources.py # Download NLTK resources
│   ├── organize_project.py    # Project structure organizer
│   └── recreate_dictionary.py # Dictionary recreation
├── train/                     # Model training files
│   ├── train.py               # English model training
│   ├── train_vietnamese.py    # Vietnamese model training
│   └── retrain.py             # Model retraining
├── quickstart.py              # Script to automate setup and run process
├── requirements.txt           # Required libraries
└── README.md                  # Documentation
```

## Quick Start

### Installation

1. Install required libraries:
```bash
pip install -r requirements.txt
```

2. Download NLTK resources (if needed):
```bash
python tools/download_nltk_resources.py
```

Or use the automatic script:
```bash
python quickstart.py --setup
```

### Using the Chatbot

The simplest way to start is using the unified chat interface:
```bash
python chat/unified_chat.py
```

This interface will automatically detect the language and use the appropriate model.

Or use the quickstart script with different modes:
```bash
python quickstart.py --chat english            # English mode
python quickstart.py --chat vietnamese_native  # Vietnamese mode (native model)
python quickstart.py --chat vietnamese_translate # Vietnamese mode (translation)
python quickstart.py --chat auto               # Auto-detect best model
```

## Detailed Usage Guide

### 1. Training Models

#### Training English Model:
```bash
python train/train.py
```

#### Training Vietnamese Model:
```bash
python train/train_vietnamese.py
```

Or use the automatic script:
```bash
python quickstart.py --train english    # Train English model
python quickstart.py --train vietnamese # Train Vietnamese model
python quickstart.py --train all        # Train both models
```

You can monitor the training progress through the displayed information, and the models will be saved in the `models/` directory.

### 2. Adding Training Data

#### English:
Add conversation pairs to the `data/conversation.txt` or `data/conversation_new.txt` file in the format:
```
English question [tab] English answer
```

#### Vietnamese:
Add conversation pairs to the `data/conversation_vietnamese.txt` file in the format:
```
Vietnamese question [tab] Vietnamese answer
```

## Model Architecture

The project uses a Sequence-to-Sequence architecture with an Encoder-Decoder mechanism:

### Encoder
The encoder takes the input sequence (user's question) and encodes it into a fixed-size representation vector:
- **Embedding Layer**: Converts each word into a feature vector
- **GRU (Gated Recurrent Unit)**: Processes the sequence of embedding vectors to create a final representation vector

### Decoder
The decoder takes the representation vector from the Encoder and generates the output sequence (answer):
- **Embedding Layer**: Converts each word into a feature vector
- **GRU**: Uses the hidden state from the Encoder to predict each word in the output sequence
- **Linear Layer**: Converts the feature vector into a probability distribution for each word in the dictionary

### Vietnamese Support
The project supports Vietnamese through two methods:

1. **Translation Method**: Converting Vietnamese characters to English, using the English model, then translating the response back to Vietnamese.

2. **Native Vietnamese Model**: Training a new model with Vietnamese data, allowing direct processing of Vietnamese characters.

## Model Customization

You can adjust parameters in the training files to improve performance:

- `EMB_DIM`: Embedding size (default: 256)
- `HID_DIM`: Hidden layer size (default: 128) 
- `BATCH_SIZE`: Batch size (default: 64)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `EPOCHS`: Number of epochs (default: 50)

## Common Error Handling

### "Token not in dictionary" Error
Occurs when the user enters a word that's not in the training dictionary. The model can only recognize words that appeared in the training data.

### "Not enough training data" Error
If you see a warning about insufficient training data, you need to add more question-answer pairs to the data file. The model needs at least 10-20 pairs to train effectively, and more data yields better results.

### "CUDA out of memory" Error
Occurs when the GPU doesn't have enough memory to train the model. Solutions:
- Reduce batch size
- Reduce model size (EMB_DIM, HID_DIM)
- Use CPU instead of GPU

## Tips for Improving Model Quality

### 1. Enhance Training Data
Add diverse question-answer pairs. Some ways to expand data:
- Add questions and answers on various topics
- Add variations of the same question (e.g., "who are you?" and "what is your identity?")
- Ensure a mix of short and long answers to help the model learn to respond diversely

### 2. Adjust Training Parameters
In the training file, you can adjust:
- `EPOCHS`: Increase to 100-200 for deeper learning
- `EMB_DIM`: Increase to 512 for better semantic understanding
- `HID_DIM`: Increase to 256 for better context memory
- `LEARNING_RATE`: Decrease to 0.0005 if the model isn't converging

## Quickstart Command Overview

The `quickstart.py` script is a tool to help you perform common tasks quickly.

### General Syntax
```bash
python quickstart.py [COMMAND]
```

### Available Commands

1. **Environment Setup**
```bash
python quickstart.py --setup
```
This command will install the necessary libraries and download NLTK resources.

2. **Model Training**
```bash
python quickstart.py --train english      # Train English model
python quickstart.py --train vietnamese   # Train Vietnamese model
python quickstart.py --train all          # Train both models
```

3. **Launch Chatbot**
```bash
python quickstart.py --chat english            # English mode
python quickstart.py --chat vietnamese_translate # Vietnamese mode (translation)
python quickstart.py --chat vietnamese_native  # Vietnamese mode (native model)
python quickstart.py --chat auto               # Auto-detect best model
```

All chat commands use the unified interface (`unified_chat.py`) with appropriate parameters.


