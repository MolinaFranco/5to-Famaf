# Claude Development Guidelines

## Code Style Requirements

-**Its NOT necesary to run flake8 in the test directory: pycloude**

- **ALWAYS run flake8 before committing any Python changes**
- Follow PEP 8 style guidelines strictly
- Maximum line length: 79 characters (flake8 default)
- Use 4 spaces for indentation (no tabs)
- **ALWAYS run flake8 after finishing the changes and fix all issues before continuing**
- Claude must **strictly respect flake8** at all times, after every modification.
- **Outside of scikit-neuromsi, it is not necessary to run flake8**

## Pre-commit Commands

Before any commit, run:
```bash
flake8 scikit-neuromsi/
```

Fix ALL flake8 issues before proceeding.

## Project Structure

- to run anything of the project you can use the virtualenv tesis with: workon tesis

- Main package: scikit-neuromsi/
- All files created for testing the changes maked, experimentation, or temporary purposes: pycloude/
- When we need to work with explicit figures taken from the paper and their necessary simulations, we will work in the directory: figures/
- figures/data: to generate the simulations and save the information
- figures/Figx: where x is a value and in this folder, the simulation of figure x from the paper will be performed by extracting information from data
- if you are testing and the tests generate outputs you have yo save them in the directory pycloude/outputs/N_name_of_the_test where N will be a index in the directory and name+of_the_test is the name of the test who made these outputs.
- Never place testing or experimental scripts inside scikit-neuromsi/.
- ssn_inference_numerical_experiments and ssn_inference_optimizer: They are reliable repositories that do not need to be modified and you can extract information and methods to use. **You should ALWAYS respect the style and architecture of scikit-neuromsi** since all our changes will be there.
- **When you need the parameters that echeveste used (Table S1) there are in parameters.md**

## Development Workflow

1. Make code changes
2. Check that everything is well documented and explained, with docstrings in English and comments referencing the papers.
3. Run flake8 to check style
4. Fix any style issues
5. Run tests

## Documentation and Comments

- **All docstrings must be complete and written in English.** Everything between ''' or """ must be in English.
- Inline comments using # must be written in Spanish.
- Docstrings must clearly explain the purpose, parameters, return values, and include references to the related papers or source code.

## Justification of Code and Methods

- All code, decisions, and implementations must be justified based on: The papers located in the repository, especially:
        echepaper (Echeveste et al., 2020)
        cupini2017 (Cupini et al., 2017)
- The original repositories already cloned and available in the local directory:
    ssn_inference_numerical_experiments
    ssn_inference_optimizer
- Claude must try to not look for external information on the internet to justify implementations.
- If more information or clarification is needed, it should ask the user directly instead.
- When using or adapting existing code, **always note the origin and reference the corresponding paper or source file.**

## Code preferences

- Please if you have to run a script that will take a long time, send it to background and review it every 1 or 2 minutes.
- In our work, each neuron represents an orientation in degrees, and it is important that we always prioritize working with degrees from -90 to 90 degrees.
- **If you have to run any script it probable that you need to run it in the virtual enviroment "tesis" with the command "workon tesis". This have all the dependencies installed**
- I prefer that you use descriptive variable names or at least explain them. Many times in papers they use only one letter, but it can be confusing in large quantities to use only one letter.
- **It is completely forbidden to use any commands on git**
- If you have to run a script that's going to take a long time, I'd prefer you wait until it finishes. Don't end the process, just stay with a "wait"
- We have a file called "parameters.md" where the necessary parameters to perform the same echeveste simulation are located.

## Figure Generation Guidelines

When generating scientific figures with multiple subplots, follow these strict visualization rules to ensure clarity and consistency:

### 1. Consistent Scales for Shared Units
- **When multiple subplots share the same physical units (e.g., mV, ms, correlation), they MUST use identical axis scales.**
- This enables direct visual comparison between subplots without mental rescaling.
- Example: If comparing "Ideal Observer: Mean $u_E$ (mV)" vs "Network: Mean $u_E$ (mV)", both Y-axes must have the same range (e.g., 0-7 mV).

### 2. Color Differentiation for Different Units
- **When changing the unit of measurement or showing a different quantity, use a distinct colormap or color scheme.**
- This provides immediate visual feedback that the data represents a different physical quantity.
- Example: If showing membrane potentials ($u_E$) with one colormap (e.g., 'RdBu_r'), use a different colormap (e.g., 'viridis' or 'plasma') for stimulus input (h) even if displayed in the same figure.

### 3. Minimize Redundant Colorbars
- **Avoid repeating colorbars unnecessarily. One colorbar per unique scale is sufficient.**
- If multiple subplots share the same colormap and scale (vmin, vmax), use a single shared colorbar.
- Example: In a figure with 10 correlation matrices all using the same scale (-1 to 1), use one or two colorbars maximum, not 10.
- Implementation options:
  - Place a single colorbar on the side of the entire figure
  - Use one colorbar per row if there are multiple rows with different ranges
  - Only repeat colorbars when scales differ between subplots

### 4. Aligned Time/Spatial Axes
- **When multiple subplots share the same time axis or spatial dimension, ensure they have identical scales and physical lengths.**
- All time axes must span the same range (e.g., 0-1000 ms) and be rendered with the same pixel width.
- This allows readers to visually align events across subplots by drawing mental vertical/horizontal lines.
- Example: If three heatmaps show activity over 0-1000 ms, all three must have the same temporal extent and tick marks.

### 5. Consistent Tick Marks
- Use consistent tick mark intervals across subplots with shared dimensions.
- Example: If one plot uses time ticks at [0, 200, 400, 600, 800, 1000] ms, all other plots with the same time range should use identical ticks.

### 6. Colorbar Label Clarity
- Always include units in colorbar labels (e.g., "$u_E$ (mV)", "h (a.u.)", "Correlation").
- Use LaTeX formatting for mathematical symbols when appropriate.
- Ensure colorbar labels are concise but informative.

### 7. Aspect Ratio Consistency
- Maintain consistent aspect ratios for subplots displaying the same type of data.
- Use `aspect='auto'` cautiously and only when necessary; prefer `aspect='equal'` for correlation matrices and spatial maps.

### Implementation Checklist

Before finalizing any multi-panel figure, verify:
- [ ] Do subplots with the same units use identical axis scales?
- [ ] Are different quantities visually distinguishable (different colormaps/colors)?
- [ ] Are colorbars minimized (no unnecessary repetition)?
- [ ] Do shared temporal/spatial axes have identical scales and lengths?
- [ ] Are tick marks consistent across related subplots?
- [ ] Do all colorbars have clear labels with units?
- [ ] Is the overall layout clean and readable?

### Examples of Good Practice

**Good**: Two heatmaps showing $u_E$ over time with:
- Same colormap ('RdBu_r')
- Same scale (vmin=0, vmax=10)
- Same time axis (0-1000 ms with ticks every 200 ms)
- Single shared colorbar

**Good**: Stimulus input (h) displayed below membrane potential ($u_E$) with:
- Different colormap to distinguish quantities
- Same time axis and width as $u_E$ plots above
- Separate colorbar with different scale appropriate for h

**Bad**: Five correlation matrices with:
- Five identical colorbars (should be one shared colorbar)
- Inconsistent axis ranges despite showing the same data type

**Bad**: Comparing "Ideal Observer" vs "Network" mean responses with:
- Ideal Observer: Y-axis 0-0.07 mV
- Network: Y-axis 0-7 mV
- (Makes direct visual comparison impossible; should use same scale)

# Final Work of Carrer (TFC) Development
- with all our work we have to write a paper in /TFC/TFC_Franco__Mateo__Molina
- we have to use mathematical and formal lenguage in spanish
- its so important that you base your rigting in others papers
- you have to use a los of conectors of sentences
- **it so importatn that you never do a circular righting** this is when you talk  about a topic and then after a while you go back to talk about the same thing.
- you have other TFC in /TFC/Tesisna_feets **it soo important that you studie that to base your work in this**
- you have to bee so academic and profesional but dont apear a robot meybe a joke could be a good option
- **IT IS EXTREMELY IMPORTANT TO EXPLAIN WHERE YOU GET ALL THE INFORMATION YOU GET FROM.**
- You can find a lot of information on Wikipedia; it's especially important for explaining historical contexts.
- It has to be like a story that follows a line and tells you how things happened over time.