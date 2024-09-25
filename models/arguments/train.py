import gc
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, PreTrainedModel, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup
from models.dataset import CustomDataset
from models.datatypes import ModelArgs
from tools.dataset.manager import DatasetManager


def load_tokenizer() -> PreTrainedTokenizerBase:
    return T5Tokenizer.from_pretrained('t5-small')


def load_model() -> PreTrainedModel:
    return T5ForConditionalGeneration.from_pretrained('t5-small')


def collect_garbage():
    gc.collect()
    torch.cuda.empty_cache()

def train(epoch, tokenizer, model, device, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for step, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if step % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {step}/{len(loader)}, Loss: {loss.item()}')

        collect_garbage()

    avg_loss = total_loss / len(loader)
    print(f'Epoch: {epoch}, Average Training Loss: {avg_loss}')


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    total_loss = 0
    predictions = []
    references = []

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            total_loss += loss.item()

            generated_ids = model.generate(input_ids=ids, attention_mask=mask, max_length=256)
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            references.extend(target)

            collect_garbage()

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(references, predictions)
    print(f'Epoch: {epoch}, Validation Loss: {avg_loss}, Accuracy: {accuracy * 100:.2f}%')
    return avg_loss, accuracy


def load_model_args() -> ModelArgs:
    model_args = ModelArgs()
    model_args.max_seq_len = 512
    model_args.max_target_len = 256
    model_args.max_batch_size = 4
    model_args.epochs = 4
    model_args.learning_rate = 1e-4

    return model_args


def load_cuda_device(model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, device


def load_dataset(tokenizer, model_args: ModelArgs):
    df = DatasetManager().read_dataset('arguments_dataset.csv')

    if df is None or df.empty:
        raise ValueError('Invalid Dataset.')

    train_size = 0.75
    train_dataset = df.sample(frac=train_size, random_state=200)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f'Treinamento: {train_dataset.shape}')
    print(f'Validação: {val_dataset.shape}')

    training_set = CustomDataset(train_dataset, tokenizer, model_args.max_seq_len, model_args.max_target_len)
    val_set = CustomDataset(val_dataset, tokenizer, model_args.max_seq_len, model_args.max_target_len)

    train_params = {'batch_size': model_args.max_batch_size, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': model_args.max_batch_size, 'shuffle': False, 'num_workers': 0}

    return DataLoader(training_set, **train_params), DataLoader(val_set, **val_params)


def main():
    tokenizer = load_tokenizer()
    model = load_model()
    model_args = load_model_args()

    model, device = load_cuda_device(model)
    train_loader, val_loader = load_dataset(tokenizer, model_args)

    optimizer = AdamW(model.parameters(), lr=model_args.learning_rate)
    total_steps = len(train_loader) * model_args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    try:
        collect_garbage()

        for epoch in range(model_args.epochs):
            train(epoch, tokenizer, model, device, train_loader, optimizer, scheduler)
            validate(epoch, tokenizer, model, device, val_loader)
    except Exception as e:
        print(e)
        print(f'{"-" * 10} Saving Model... {"-" * 10}')

    model.save_pretrained('./t5_finetuned_model')
    tokenizer.save_pretrained('./t5_finetuned_tokenizer')


if __name__ == '__main__':
    main()
