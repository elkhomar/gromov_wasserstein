Compute the third lower bound (it is like GW but easier since we first compute Lp and run the integration on only 2 vars instead of 4)( hierarchical transport),

The tlb looks like Gilmore-Lawler lower bound in the QAP . P.M. Pardalos, H. Wolkowicz (eds.), Quadratic Assignment and Related Problems, from page 17
It could be interesting to compute the QAP lower bound instead of the ot one ? Compare Mémoli and Gilmore-Lawler formulations and see if the gl

!Care what should converge to what, the tlb is on the gw distance not the entropic gw nor the entropic gw estimator !

Run the entropy regularized gw algo that is initiallized at random (random initial coupling)
Then run it but starting from the coupling found in the third lower bound (to what accuracy should the tlb estimator go ?)


run on : (2D/ 3D gaussians), shapes for which we know the theoretical result of gw (random shapes that are changed through an isometry, add quantifiable noise to the shapes)