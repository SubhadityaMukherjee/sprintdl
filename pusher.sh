black "."
isort .
pdoc --force --html -o docs sprintdl
mv docs/sprintdl/index.html docs/index.md
mv docs/sprintdl/* docs/
jupytext --to script "demo.ipynb"
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
