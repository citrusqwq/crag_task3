import sys
from transformers import T5EncoderModel

in_path = sys.argv[1]
out_path = sys.argv[2]

model = T5EncoderModel.from_pretrained(in_path)
model.half()
model.save_pretrained(out_path)