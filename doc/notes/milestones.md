Big goals:
* We want manifolds that
  * Reflect delay space (via curvature)
    * Which version of curvature works best? Should rely only on latency
  * Show both global and local structure
    * This means that we need some sort of stability in our output
    * The disappearance of a less significant link should effect a relatively small change in the manifold. For larger changes, we can rely on postprocessing
* We want layers
  * Delays should be shown in the manifold's shape
  * Throughputs should be shown by an additional layer.
    * Arcs with heights are promising
    * Style of the arcs should be chosen
* We want animation
  * The webapp has several rendering bugs that should be fixed
  * A link going down should have some sort of extra indication

Plan:
* 04/07: Grab data from ESnet.
* 04/09: Generate network graphs for ESnet datasets.
* 04/11: Generate manifolds (naively) from the ESnet datasets. For this milestone, don't focus on getting the manifolds looking perfect. Instead, we just want some sort of identifiable features, as well as reaching the MVP stage with real world data.
* 04/14: Generate toy examples based on the ESnet data. Generate some manifolds for these toy examples.
* 04/14: Port over the current Matisse manuscript to the new LaTeX template. Figure out how many pages we need to cut.
* 04/21: Make revisions to the toy examples and form a full "dictionary" of features we want
* 04/28: Regenerate manifolds using the real world data, using what we learned from the toy examples.
* 05/05: Determine where exactly we need to pare down the current MASCOTS submission.
* 05/05: Create a LaTeX template for the HotNets manuscript (incl. determining the overall structure of the paper).
* 05/05: Put together the bibliography for the Related Works section of the HotNets paper.
* 05/05: Polish the webapp's UI (color choices, etc.).
* 05/12: Write the initial draft of the HotNets paper.
* 05/18: Submit the MASCOTS paper.
* 05/29: Register the HotNets paper.
* 06/05: Submit the HotNets paper.
