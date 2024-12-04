# Background and prerequisites

## Radio astronomy

A background in radio astronomy, Fourier transforms, and interferometry is a prerequisite for using MPoL but is beyond the scope of this documentation. We recommend reviewing these resources as needed.

- [Essential radio astronomy](https://www.cv.nrao.edu/~sransom/web/xxx.html) textbook by James Condon and Scott Ransom, and in particular, Chapter 3.7 on Radio Interferometry.
- NRAO's [17th Synthesis Imaging Workshop](http://www.cvent.com/events/virtual-17th-synthesis-imaging-workshop/agenda-0d59eb6cd1474978bce811194b2ff961.aspx) recorded lectures and slides available
- [Interferometry and Synthesis in Radio Astronomy](https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract) by Thompson, Moran, and Swenson. An excellent and comprehensive reference on all things interferometry.
- The [Revisiting the radio interferometer measurement equation](https://ui.adsabs.harvard.edu/abs/2011A%26A...527A.106S/abstract) series by O. Smirnov, 2011
- Ian Czekala's lecture notes on [Radio Interferometry and Imaging](https://iancze.github.io/courses/as5003/lectures/)

RML imaging is different from CLEAN imaging, which operates as a deconvolution procedure in the image plane. However, CLEAN is by far the dominant algorithm used to synthesize images from interferometric data at sub-mm and radio wavelengths, and it is useful to have at least a basic understanding of how it works. We recommend

- [Interferometry and Synthesis in Radio Astronomy](https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract) Chapter 11.1
- David Wilner's lecture on [Imaging and Deconvolution in Radio Astronomy](https://www.youtube.com/watch?v=mRUZ9eckHZg)
- For a discussion on using both CLEAN and RML techniques to robustly interpret kinematic data of protoplanetary disks, see Section 3 of [Visualizing the Kinematics of Planet Formation](https://ui.adsabs.harvard.edu/abs/2020arXiv200904345D/abstract) by The Disk Dynamics Collaboration

## Statistics and Machine Learning

MPoL is built on top of the [PyTorch](https://pytorch.org/) machine learning framework and adopts much of the terminology and design principles of machine learning workflows. As a prerequisite, we recommend at least a basic understanding of statistics and machine learning principles. Two excellent (free) textbooks are

- [Dive into Deep Learning](https://d2l.ai/), in particular chapters 1 - 3 to cover the basics of forward models, automatic differentiation, and optimization.
- [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) for a lengthier discussion of these concepts and other foundational statistical concepts.

And we highly recommend the informative and entertaining 3b1b lectures on [deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown).

## PyTorch

As a PyTorch library, MPoL expects that the user will write Python code that uses MPoL primitives as building blocks to solve their interferometric imaging workflow, much the same way the artificial intelligence community writes Python code that uses PyTorch layers to implement new neural network architectures (for [example](https://github.com/pytorch/examples)). You will find MPoL easiest to use if you follow PyTorch customs and idioms, e.g., feed-forward neural networks, data storage, GPU acceleration, and train/test optimization loops. Therefore, a basic familiarity with PyTorch is considered a prerequisite for MPoL.

If you are new to PyTorch, we recommend starting with the official [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) guide. You can also find high quality introductions on YouTube and in textbooks.

## RML Imaging

MPoL is a modern PyTorch imaging library, however many of the key concepts behind Regularized Maximum Likelihood image have been around for some time. We recommend checking out the following (non-exhaustive) list of resources

- [Regularized Maximum Likelihood Image Synthesis and Validation for ALMA Continuum Observations of Protoplanetary Disks](https://ui.adsabs.harvard.edu/abs/2023PASP..135f4503Z/abstract) by Zawadzki et al. 2023
- The fourth paper in the 2019 [Event Horizon Telescope Collaboration series](https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract) describing the imaging principles
- [Maximum entropy image restoration in astronomy](https://ui.adsabs.harvard.edu/abs/1986ARA%26A..24..127N/abstract) AR&A by Narayan and Nityananda 1986
- [Multi-GPU maximum entropy image synthesis for radio astronomy](https://ui.adsabs.harvard.edu/abs/2018A%26C....22...16C/abstract) by Cárcamo et al. 2018
- Dr. Katie Bouman's Ph.D. thesis ["Extreme Imaging via Physical Model Inversion: Seeing Around Corners and Imaging Black Holes"](https://people.csail.mit.edu/klbouman/pw/papers_and_presentations/thesis.pdf)
