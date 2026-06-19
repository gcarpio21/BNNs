
# Report

How to compile:

	pdflatex template --shell-escape
	bibtex template
	pdflatex template --shell-escape
	pdflatex template --shell-escape
	
The ```--shell-escape``` option is used to externalize latex/tikz figures.
This heavily speeds up the compile time if you have many tikz figures in your report.

If you are working with Overleaf, you can specify a custom `latexmkrc` file:
- https://www.overleaf.com/learn/latex/Articles/How_to_use_latexmkrc_with_Overleaf

## ```matlab2tikz```

We have a script ```matlab2tikz``` that exports your MATLAB figures to latex/tikz.
You can find an example in the report.
If you want to use it as well, ask your advisor for access:
https://gitlab.lrz.de/cora/matlab2tikz.

Clone the repository and add it to the matlab path.
To create a tikz figure, save your figure, and convert it to tikz using the `convertToTikz.m` matlab file.

	% create your figure -------------------------------------------------
	
	% ...
	
	figure; hold on; box on
    useCORAcolors("CORA:contDynamics")
    projDims = [1,2];

    % plot reachable sets
	% for reachable sets, you can use the 'Unify' option to speed up plotting and the export to tikz 
	% by unifying the computed time interval solutions before plotting.
	% for complex sets, you can increase the total number of sets such that the set is unified in 'n' parts.
    plot(R,projDims, 'DisplayName', 'Reachable set','Unify',true,'UnifyTotalSets',1);

    % plot initial set
    plot(R.R0,projDims,'DisplayName','Initial set');

    % plot simulation results
    plot(simRes,projDims, 'DisplayName', 'Simulations');

    % label plot
    xlabel(['x_{',num2str(projDims(1)),'}']);
    ylabel(['x_{',num2str(projDims(2)),'}']);
    legend('Location', 'northwest')
	
	% export to tikz -----------------------------------------------------
	
	figpath = 'myplot.fig';
	savefig(figpath);
	convertToTikz(figpath);
	
This should give you a `myplot.tikz` file.
Copy it to your latex folder under `./figures`.
You can then add it to the report using the following code.
Don't forget to enable `--shell-escape` (see above):

	\begin{figure}[t]
	  \begin{center}
	    \includetikz{./figures/myplot} % without file extension!
	  \end{center}
	  \caption{A vector graphic loaded from a tikz file}
	  \label{Pic4}
	\end{figure}
	
If you have a plot with multiple subplots, you get one tikz file per subplot, a main tikz file and legend tikz file.
Copy all of them to your latex folder and adjust the `\basepath` variable:

	\def\basepath{./figures/}