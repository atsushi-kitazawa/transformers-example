import torch
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM


def japanese():
    # トークナイザーとモデルの準備
    tokenizer = AutoTokenizer.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    model = AutoModelForMaskedLM.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')

    # 入力テキストのエンコード
    # sentence = f'吾輩は{tokenizer.mask_token}である。名前はまだ無い。'
    sentence = f'私は30代男性です。休日は{tokenizer.mask_token}します'
    input_ids = tokenizer.encode(
        sentence, return_tensors='pt')
    print('input_ids:', tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))

    # マスクインデックスの取得
    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[
        1].tolist()[0]
    print('masked_index:', masked_index)

    # マスクトークンの予測
    result = model(input_ids)
    pred_ids = result[0][:, masked_index].topk(5).indices.tolist()[0]
    for pred_id in pred_ids:
        output_ids = input_ids.tolist()[0]
        output_ids[masked_index] = pred_id
        print(tokenizer.decode(output_ids))

def japanese2():
    tokenizer = AutoTokenizer.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    model = AutoModelForMaskedLM.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    nlp = pipeline(task='fill-mask', tokenizer=tokenizer, model=model)
    sentence = f'吾輩は{tokenizer.mask_token}である。名前はまだ無い。'
    result = nlp(sentence)
    for r in result:
        print("score:{}, token:{}".format(r['score'], r['token_str']))

if __name__ == "__main__":
    # english()
    # japanese()
    japanese2()
