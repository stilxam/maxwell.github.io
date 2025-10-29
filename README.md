<div align="center">
Maxwell Litsios | BSc | MSc Student | Eindhoven, NL | +41767996053 | <a href="https://www.linkedin.com/in/maxwell-litsios-hilber">LinkedIn</a> | <a href="https://github.com/stilxam">GitHub</a> | <a href="maxwell.litsios.hilber@gmail.com">e-mail</a>
</div>

---

## About me
I thrive in collaborative environments, using machine learning to help people make smarter decisions and eliminate tedious processes. This drives my desire to delve deeper into the learning dynamics of neural networks, specifically exploring how to reduce the computational resources required for model development. Outside of work, I compete in strongman and coach Olympic weightlifting.

## Professional Experience

### IBI Benchmarking | Data Scientist & Software Engineer
[2023-2024] Eindhoven, NL 
*   Architected and co-led the successful migration of the company's entire codebase from C# to Python, improving maintainability.
*   Oversaw the expansion of the data science team from 3 to 6 members, managing the full recruitment cycle from interviews to onboarding into our SCRUM workflow.
*   Designed and implemented novel rating methodologies for asset diversification, providing institutional investors with enhanced data-driven insights.
*   Architected the data model and strategic plan for a new European pension database.

### Eindhoven University of Technology | Teaching Assistant :
<details>
<summary>
[2023-now] - Programming for Data Science [JBI010] and for Applied Physics [31PAP]:
</summary>
<ul>
<li> Mentored and instructed classes of 60 students in Python, created auxiliary course material, graded assignments, invigilated exams, and led course material recaps during instructor absences.
</ul>
</details>
<br/>

<details>
<summary>
[2023-now] - Project Supervisor [JBG030/JBG040]:
</summary>
<ul>
 <li>Supervised teams of 6–8 students in developing data-driven, explainable AI methods for predictive policing, ensuring projects met addressed stakeholder concerns, technical requirements and ethical considerations.
</ul>
</details>


## Education
###  [2025-now] Masters in Data Science and AI -- Eindhoven University of Technology

#### Uncertainy-Aware Budget Fitted Q-learning (ongoing)
<details>
<summary>Details</summary>
<ul>
<li>Developing multiple deep reinforcement learning agents to navigate complex driving scenarios while adhering to safety constraints using a Budgeted Q-learning framework.</li>
<li>Implemented Bayesian Neural Network (BNN) with Pyro, Monte Carlo Dropout and Deep Ensembles, to model probabilistic distributions over Q-values, quantifying model uncertainty to enable risk-averse decision-making.</li>
</ul>
</details>
<br/>

#### Surveying the Field Of Implicit Neural Representations (ongoing)
<details>
<summary>Details</summary>
This is a continuation of my Bachelor End Project, I am implementing methods such as mixed precision training, <a href="https://github.com/ubc-vision/nf-soft-mining/tree/main">Soft-Mining</a> and <a href="https://github.com/NVlabs/instant-ngp/tree/master/src">Instant-Neural-Graphic Primitives</a> to accelerate training of Implicit Neural Representations in order to finish running the experiments for a survey of methods to learn Implicit Neural Representations (INRs) e.g. NeRFs, SDFs.
</details>
<br/>

### [2021-2025] Bachelors in Data Science -- Eindhoven University of Technology and University of Tilburg
#### Bachelor End Project: Neural Tangent Kernels To Measure Spectral Bias In INRs (9.5/10)

<details>
<summary>Details</summary>
<ul>

<li> Built an <a href="https://github.com/stilxam/INR_BEP">open-source INR benchmarking codebase</a>
<li> Implemented 9 layer types and 2 positional encodings.
<li> Added suport to learn fourier transforms, derivatives and integral representations.
<li> Developed a NTK based trainability metric for INRs, outperforming the standard condition-number measure.
</ul>
</details>
<br/>


### Project Highlights

**Mamba Generative Code Infilling with Masked Diffusion (ongoing)**<br>
**Why?** - I am frustrated with the quality of jax code suggested by autocomplete models.
<details>
<summary>Technical Details</summary>
<p>
<em>Stack:</em> <b>JAX, Equinox, jaxtyping, Optax, Hugging Face Transformers, orbax</b>
<ul>
    <li>Architected and training a novel, small (~40M params) generative model for JAX code, to fill partially masked sequences.</li>
    <li>Designed a hybrid transformer architecture that fuses a Bidirectional MAMBA (SSM) for efficient global context modeling with local Multihead Attention for capturing fine-grained syntactical patterns.</li>
    <li>Engineered a comprehensive data pipeline from scratch: scraped 100M+ tokens of relevant JAX-focused GitHub repositories, implemented a robust cleaning and filtering script, and trained a custom 50,000-token Byte-Level BPE tokenizer.</li>
    <li>Implemented a training loop in JAX/Equinox featuring mixed-precision (fp16), dynamic loss scaling, and gradient accumulation to train with a larger effective batch size of on a single accelerator.</li>
</ul>
</p>
</details>
<br/>

**EquiNTK: Neural Tangent Kernel (NTK) in JAX**<br>
**Why?** - Google's <a href="https://github.com/google/neural-tangents">neural-tangents</a> got archived, it was buggy and did not offer first party support to equinox models.
<details>
<summary>Technical Details</summary>
<p>
<em>Stack:</em> <b>JAX, Equinox, jaxtyping</b>
<ul>
    <li>Developed a library for theoretical analysis of deep neural networks, implementing the Neural Tangent Kernel (NTK) and Neural Network Gaussian Process (NNGP) from first principles.</li>
    <li>Engineered memory-efficient NTK computation leveraging JAX primitives to process large datasets in batches, avoiding common out-of-memory errors.</li>
    <li><b>Experimental:</b> a greedy function to derive "learning-difficulty" for each data pair to offer a "smart" sampling method to address neural networks' spectral bias.</li>
    <li>Built a predictive module to forecast a model's training trajectory on new data, effectively using kernel regression to solve the linear dynamics described by the NTK.</li>
    <li>Designed the library to robustly handle models with multi-dimensional outputs.</li>
</ul>
</p>
</details>
<br/>

**GOLF: Gradient-Optimized piecewise Linear Function in JAX**<br>
**Why?** - I wanted to fit 100 000+ piecewise continuous linear functions, current solutions (<a href="https://pypi.org/project/pwlf/">PWLF</a>) could not be run in parallel and did not leverage GPU acceleration.
<details>
<summary>Technical Details</summary>
<p>
<em>Stack:</em> <b>JAX, Equinox, Optax</b>
<ul>
    <li>Developed a fully-differentiable piecewise linear spline model from scratch using JAX and Equinox, enabling end-to-end gradient-based optimization of both breakpoint positions and values.</li>
    <li><b>Experimental:</b> Designed and implemented a custom L1 regularization loss term on slope changes, which encourages model simplicity by automatically merging and removing unnecessary linear segments during training.</li>
    <li>Engineered an initialization technique that places initial breakpoints in regions of high data curvature, leading to significantly faster convergence and superior final model accuracy.</li>
</ul>
</p>
</details>
<br/>

**EquiMixedP: Mixed-Precision Training for Equinox**<br>
**Why?** - While working on the masked diffusion language model reference above, the loss scaling in <a href="https://github.com/google-deepmind/jmp">jmp</a> was not interacting nicely with JAX's JIT compiler and <a href="https://github.com/Data-Science-in-Mechanical-Engineering/mixed_precision_for_JAX/tree/main/mpx">mxp</a> was buggy, hence, I reimplemented it to allow for easy mixed precision JIT compiled training with equinox modules.
<details>
<summary>Technical Details</summary>
<p>
<em>Stack:</em> <b>JAX, Equinox, Optax</b>
<ul>
    <li>Implemented a library to enable mixed-precision training (<code>bfloat16</code>/<code>float16</code>) for Equinox models, reducing memory consumption and accelerating training.</li>
    <li>Made a dynamic loss scaling mechanism to automatically prevent numerical underflow and overflow, ensuring stable training with half-precision data types.</li>
    <li>Designed a user-friendly API with a high-level <code>MixedPrecision</code> manager class and custom gradient wrappers that handle casting, loss scaling, and unscaling within the JAX pipeline.</li>
</ul>
</p>
</details>
<br/>

## Technical Skills
*   **Languages:** Python, Nix, R, SQL, C++
*   **ML & Data Science:** JAX, Equinox, Optax, Orbax, RLax, Gym, NumPyro, PyTorch, NumPy, Polars, Pandas, SciPy, sk-learn
*   **Tools & Platforms:** NixOS, Linux, Git, duckdb
*   **Visualization:** Matplotlib, Seaborn, Plotly, Dash, Bokeh
*   **Data & Modeling:** Relational Algebra, E-R Modeling, Datalog

## Awards
*   Volunteer of the Year, Eindhoven Student Sport Federation (2024-2025)
*   Best Project Award (out of 27 teams) for "Police Force Resource Optimization" presented to the London Metropolitan Police.
*   Best Pitch Award (out of 30 teams) for "Pitch Competition - Data Driven Business Creations" presented to a set of industry experts at [JADS](https://www.jads.nl/)

## Languages
| French | English | Spanish | German | Dutch |
| :---: | :---: | :---: | :---: | :---: |
| Native | Native | Native | Fluent | Proficient (Reading & Listening) |

