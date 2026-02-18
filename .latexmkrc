# Always rebuild if something smells stale
$force_mode = 1;

# Track file dependencies aggressively
$recorder = 1;

# Use pdflatex in nonstop mode
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Watch all included files
$dependents_phony = 1;
