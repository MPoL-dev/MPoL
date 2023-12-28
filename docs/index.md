---
substitutions:
  Discussions badge: |-
    ```{image} https://img.shields.io/badge/community-Github%20Discussions-orange
    :target: https://github.com/MPoL-dev/MPoL/discussions
    ```
  Tests badge: |-
    ```{image} https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml
    ```
---

# Million Points of Light (MPoL)

{{ Tests badge }}
{{ Discussions badge }}

```{raw} html
<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/MPoL-dev/MPoL" data-color-scheme="no-preference: light; light: light; dark: dark_dimmed;" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star MPoL-dev/MPoL on GitHub">Star</a>
<a class="github-button" href="https://github.com/MPoL-dev/MPoL/discussions" data-color-scheme="no-preference: light; light: light; dark: dark_dimmed;" data-icon="octicon-comment-discussion" data-size="large" aria-label="Discuss MPoL-dev/MPoL on GitHub">Discuss</a>
```

MPoL is a Python framework for Regularized Maximum Likelihood (RML) imaging. It is built on top of PyTorch, which provides state of the art auto-differentiation capabilities and optimizers. We focus on supporting continuum and spectral line observations from interferometers like the Atacama Large Millimeter/Submillimeter Array (ALMA) and the Karl G. Jansky Very Large Array (VLA). There is potential to extend the package to work on other Fourier reconstruction problems like sparse aperture masking and other forms of optical interferometry.

To get a sense of how MPoL works, please take a look at the {ref}`rml-intro-label` and then the tutorials down below. If you have any questions, please join us on our [Github discussions page](https://github.com/MPoL-dev/MPoL/discussions).

If you'd like to help build the MPoL package, please check out the {ref}`developer-documentation-label` to get started. For more information about the constellation of packages supporting RML imaging and modeling, check out the MPoL-dev organization [website](https://mpol-dev.github.io/) and [github](https://github.com/MPoL-dev) repository hosting the source code.

*If you use MPoL in your research, please cite us!* See <https://github.com/MPoL-dev/MPoL#citation> for the citation.

```{toctree}
:caption: User Guide
:maxdepth: 2

rml_intro.md
installation.md
units-and-conventions.md
developer-documentation.md
api.rst
```

```{toctree}
:hidden: true

changelog.md
```

```{toctree}
:caption: Tutorials
:maxdepth: 2

ci-tutorials/PyTorch
ci-tutorials/gridder
ci-tutorials/optimization
ci-tutorials/loose-visibilities
ci-tutorials/crossvalidation
ci-tutorials/gpu_setup.rst
ci-tutorials/initializedirtyimage
large-tutorials/HD143006_part_1
large-tutorials/HD143006_part_2
ci-tutorials/fakedata
large-tutorials/pyro
```

- {ref}`genindex`
- {ref}`changelog-reference-label`
