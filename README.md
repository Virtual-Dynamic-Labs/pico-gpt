# Pico GPT

This is a Pico GPT library for the Virtual Dyno Human.

## Usage

```c
pip install pico-gpt
```

```python
from pico_gpt import PicoGPT

gpt = PicoGPT()
gpt.load_model("pico-gpt-3B.bin")

print(gpt.generate_text("Hello, my name is", 10))
```

## Training

```bash
pip install -r requirements.txt
python train.py
```

## License

MIT License
