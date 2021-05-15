black "."
isort .
pdoc --force --html -o docs sprintdl
python runner.py -f "syntax.txt" -o "syntax.pdf" -c -n "Subhaditya Mukherjee" -t "sprintDL" -w True
mv docs/sprintdl/index.html docs/index.md
mv docs/sprintdl/* docs/
for i in $(exa demos); do jupytext --to script "demos/$i"; done;
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
