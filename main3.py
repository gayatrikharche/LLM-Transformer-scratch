import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from transformer3 import LanguageModel, Encoder
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from tqdm import tqdm
import json
from utilities import Utilities

seed = 42
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch, shuffle=True)

    vocab_size= tokenizer.vocab_size
    classifier = Encoder(vocab_size, n_embd, n_layer, n_head, block_size, 0.2, n_hidden, n_output)
    classifier = classifier.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    
    for epoch in range(epochs_CLS):
        epoch_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            prediction, _ = classifier(xb)
            loss = torch.nn.functional.cross_entropy(prediction, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = compute_classifier_accuracy(classifier, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
        print(f"Epoch: {epoch}, Loss: {loss.item() :.4f}, Train Acc: {train_accuracy :.4f},Test Accuracy: {test_accuracy :.4f}" )
        print()


    sentence1 = "Only if our politics better reflects the decency of our people."
    sentence2 = "The top 10 percent of earners now take in half of all income in the U.S."
    sanity_checker = Utilities(tokenizer, classifier)
    sanity_checker.sanity_check(sentence1, block_size, file="Part3a")
    sanity_checker.sanity_check(sentence2, block_size, file="Part3b")
    print()

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    inputfile2 = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile2, 'r', encoding='utf-8') as f:
        lmHbushText = f.read()
    test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, lmHbushText,  block_size)
    test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=True)

    inputfile3 = "speechesdataset/test_LM_obama.txt"
    with open(inputfile3, 'r', encoding='utf-8') as f:
        lmObamaText = f.read()
    test_LM_obama_dataset = LanguageModelingDataset(tokenizer, lmObamaText,  block_size)
    test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=True)

    inputfile4 = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile4, 'r', encoding='utf-8') as f:
        lmWbushText = f.read()
    test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, lmWbushText,  block_size)
    test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=True)

    vocab_size= tokenizer.vocab_size
    lm= LanguageModel(vocab_size, n_embd, n_layer, n_head, block_size, 0.2)
    lm = lm.to(device)
    optimizer = torch.optim.AdamW(lm.parameters(), lr=learning_rate)
    
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i > max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        loss, _ = lm(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i%eval_interval == 0:
            train_per = compute_perplexity(lm, train_LM_loader, eval_iters)
            print(f"Iterations: {i}, Loss: {loss},Train Perplexity: {train_per:.4f}" )
            print( )
    test_ob_per=compute_perplexity(lm, test_LM_obama_loader, eval_iters)
    print(f"Iterations: {i-1}, Loss: {loss},Test Perplexity(Obama): {test_ob_per:.4f}" )
    print( )
    test_hb_per=compute_perplexity(lm, test_LM_hbush_loader, eval_iters)
    print(f"Iterations: {i-1}, Loss: {loss},Test Perplexity(Hbush): {test_hb_per:.4f}" )
    print( )
    test_wb_per=compute_perplexity(lm, test_LM_wbush_loader, eval_iters)
    print(f"Iterations: {i-1}, Loss: {loss},Test Perplexity(Wbush): {test_wb_per:.4f}" )
    print( )

    sentence1 = "Only if our politics better reflects the decency of our people."
    sentence2 = "The top 10 percent of earners now take in half of all income in the U.S."
    sanity_checker = Utilities(tokenizer, lm)
    sanity_checker.sanity_check(sentence1, block_size, file="Part3c")
    sanity_checker.sanity_check(sentence2, block_size, file="Part3d")
    print()


if __name__ == "__main__":
    main()
