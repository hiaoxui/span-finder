from argparse import  ArgumentParser

from sftp import SpanPredictor


parser = ArgumentParser('predict spans')
parser.add_argument(
    '-m', help='model path', type=str, default='https://gqin.top/sftp-fn'
)
args = parser.parse_args()

# Specify the path to the model and the device that the model resides.
# Here we use -1 device, which indicates CPU.
predictor = SpanPredictor.from_path(
    args.m,
    cuda_device=-1,
)

# Input sentence could be a string. It will be tokenized by SpacyTokenizer, and the tokens will be returned
# along with the predictions.
input1 = "Bob saw Alice eating an apple."
print("Example 1 with input:", input1)
output1 = predictor.predict_sentence(input1)
output1.span.tree(output1.sentence)

# Input sentence might already be tokenized. In this situation, we'll respect the tokenization.
# The output will be based on the given tokens.
input2 = ["Bob", "saw", "Alice", "eating", "an", "apple", "."]
print('-'*20+"\nExample 2 with input:", input2)
output2 = predictor.predict_sentence(input2)
output2.span.tree(output2.sentence)

# To be efficient, you can input all the sentences as a whole.
# Note: The predictor will do batching itself.
# Instead of specifying the batch size, you should specify `max_tokens`, which
# indicates the maximum tokens that could be put into one batch.
# The predictor will dynamically batch the input sentences efficiently,
# and the outputs will be in the same order as the inputs.
output3 = predictor.predict_batch_sentences([input1, input2], max_tokens=512, progress=True)
print('-'*20+"\nExample 3 with both inputs:")
for i in range(2):
    output3[i].span.tree(output3[i].sentence)

# For SRL, we can limit the decoding depth if we only need the events prediction. (save 13% time)
# And can possibly limit #spans to speedup.
predictor.economize(max_decoding_spans=20, max_recursion_depth=1)
output4 = predictor.predict_batch_sentences([input2], max_tokens=512)
print('-'*20+"\nExample 4 with input:", input2)
output4[0].span.tree(output4[0].sentence)
