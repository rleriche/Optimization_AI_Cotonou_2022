# Optimization_AI_Cotonou_2022

This directory contains files related to the AI summer school in Cotonou (actually Godomey) 2022, EEIA 2022.

### Conference_Optimization_Gal

Slides of a conference presentation entitled "Optimization for quantitative decisions: a versatile multi-tasker or an utopia?".

Abstract :
The study of optimization algorithms started at the end of World War II and has since then experienced a constantly growing interest, fueled by needs in
engineering, computational simulation and Machine Learning. 
In this talk, we look into the history of the new scientific objects that are the optimizers. The point-of-view is historical and philosophical rather than mathematical. 
It is explained how the conditions for the emergence of the optimizers as new scientific objects correspond to a state of sufficient knowledge decomposition. The versatility of optimizers is illustrated through examples. Although they are just tools, optimizers are a source a fascination because they guide through a space of representations in a process that resembles learning. But rationality is bounded and, in that sense, optimization problems are utopias: they are an over-simplification of decisions which are actually rooted in human relationships; they often cannot be solved because the computational capacities are limited.
The presentation finishes with some contemporary challenges in optimization.

The presentation is also available on HAL archives. Please cite as:
Rodolphe Le Riche. Optimization for quantitative decisions: a versatile multi-tasker or an utopia?. Ecole d'Ete en Intelligence Artificielle (EEIA 2022), fondation Vallet, Jul 2022, Godomey, Benin. HAL report no. hal-03725318, [slides on HAL](https://hal.archives-ouvertes.fr/hal-03725318).

### Slides_course

Slides of the course entitled "An introduction to optimization for machine learning".

### Code

Code in python

### setup your environment


7. Move to the cloned repository `cd <foldername>`
6. Run the command `python3 -m venv venv`
6. Activate the venv: `source venv/bin/activate`
6. Upgrade pip if needed `pip install --upgrade pip`
7. Install packages needed with `pip install -r requirements/dev.txt`
8. Install our package `<packagename>` with `pip install -e .`
9. Create a jupyter kernel linked to this venv 

 `ipython kernel install --name "venv" --user`

  References: https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove.

14. You can now open a notebook by running `jupyter notebook --no-browser`
14. Select the kernel `venv` to run the notebook.