# def generate_translation(model, tokenizer, example):
#     """print out the source, target and predicted raw text."""
#     source = example[source_lang]
#     target = example[target_lang]
#     input_ids = tokenizer(source)['input_ids']
#     input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
#     generated_ids = model.generate(input_ids)
#     prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

#     print('source: ', source)
#     print('target: ', target)
#     print('prediction: ', prediction)

def generate_translation(model, tokenizer, example):
    """print out the source, target and predicted raw text."""
    source = example[source_lang]
    target = example[target_lang]
    input_ids = example['input_ids']
    input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
    print('input_ids: ', input_ids)
    generated_ids = model.generate(input_ids)
    print('generated_ids: ', generated_ids)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print('source: ', source)
    print('target: ', target)
    print('prediction: ', prediction)

def generate_translation_testset(model, tokenizer, data_loader):
    start = time.time()
    rows = []
    for example in data_loader:
        input_ids = example['input_ids']
        # input_ids = torch.LongTensor(input_ids).to(model.device)
        input_ids = torch.LongTensor([item.cpu().detach().numpy() for item in input_ids]).to(model.device)
        # print('input_ids: ', input_ids)
        generated_ids = model.generate(input_ids)
        generated_ids = generated_ids.detach().cpu().numpy()
        predictions = tokenizer.decode_batch(generated_ids)

        labels = example['labels'].detach().cpu().numpy()
        targets = tokenizer.decode_batch(labels)
        for target, prediction in zip(targets, predictions):
            print("target:", target)
            print("prediction:", prediction)
            

    #     sources = source_tokenizer.decode_batch(input_ids.detach().cpu().numpy())
    #     for source, target, prediction in zip(sources, targets, predictions):
    #         row = [source, target, prediction]
    #         rows.append(row)

    # end = time.time()
    # print('elapsed: ', end - start)
    # df_rows = pd.DataFrame(rows, columns=['source', 'target', 'prediction'])




def batch_tokenize_fn(examples):
    """
    Generate the input_ids and labels field for huggingface dataset/dataset dict.
    
    Truncation is enabled, so we cap the sentence to the max length, padding will be done later
    in a data collator, so pad examples to the longest length in the batch and not the whole dataset.
    """
    sources = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = pretrained_tokenizer_eu(sources, max_length=max_source_length, truncation=True, add_special_tokens=False)
    n = len(model_inputs['input_ids'])
    # for i in range(n):
    #     model_inputs['input_ids'][i] += orig_emb_size
    #     model_inputs['input_ids'][i].append(0)
    model_inputs = pretrained_tokenizer_eu(sources, max_length=max_source_length, padding='max_length', truncation=True, add_special_tokens=False)
    for i in range(n):
        model_inputs['input_ids'][i] += orig_emb_size
    if n != max_source_length:
        model_inputs['input_ids'][n] = 0
    else:
        model_inputs['input_ids'][n-1] = 0
        
    for i in range(n + 1, len(model_inputs['input_ids'])):
        model_inputs['input_ids'][i] = 58100
    model_inputs.pop('token_type_ids')

    
    
    # pretrained_tokenizer_eu.convert_ids_to_tokens(model_inputs['input_ids'])

    # setup the tokenizer for targets,
    # huggingface expects the target tokenized ids to be stored in the labels field
    with pretrained_tokenizer.as_target_tokenizer():
        labels = pretrained_tokenizer(targets, max_length=max_target_length, truncation=True)
        n = len(labels['input_ids'])
        labels = pretrained_tokenizer(targets, max_length=max_target_length, padding='max_length', truncation=True)
        for i in range(n, max_target_length):
            labels['input_ids'][i] = -100
    # pretrained_tokenizer.convert_ids_to_tokens(labels['input_ids'])

    model_inputs["labels"] = labels["input_ids"]
    # model_inputs["input_ids"] = torch.LongTensor(model_inputs["input_ids"])
    # model_inputs["attention_mask"] = torch.LongTensor(model_inputs["attention_mask"])
    # model_inputs["labels"] = torch.LongTensor(model_inputs["labels"])
    return model_inputs