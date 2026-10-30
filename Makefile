# Keep slide content in slides.md and shared styling in hadron.css.
# The browser fetches reveal.js from the unpkg URL pinned in slides.md.
.PHONY: all clean
all: talk.html

talk.html: slides.md hadron.css hadron-logo.png llm-next-token-prediction.png sft_before.png sft_after_training.png transformer-hidden-state.svg embedding-similarity-training.svg ai_intro.ogg
	pandoc --standalone --to=revealjs --slide-level=1 -V autoPlayMedia=true slides.md -o $@

clean:
	rm -f talk.html
