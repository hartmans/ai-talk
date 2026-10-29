# Keep slide content in slides.md and shared styling in hadron.css.
# Pandoc needs network access to embed reveal.js on a fresh build.
.PHONY: all clean
all: talk.html

talk.html: slides.md hadron.css hadron-logo.png llm-next-token-prediction.png
	pandoc --standalone --to=revealjs --slide-level=1 --embed-resources slides.md -o $@

clean:
	rm -f talk.html
