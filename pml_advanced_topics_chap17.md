Probabilistic Machine Learning

| Adaptive   | Computation    | and Machine  | Learning  |
| ---------- | -------------- | ------------ | --------- |
| Francis    | Bach, editor   |              |           |
| A complete | list of titles | can be found | online at |
https://mitpress.mit.edu/search-result-list/
?series=adaptive-computation-and-machine-learning-series.

| Probabilistic   | Machine | Learning |
| --------------- | ------- | -------- |
| Advanced Topics |         |          |
Kevin P. Murphy
The MIT Press
Cambridge, Massachusetts
London, England

© 2023 Kevin P. Murphy
This work is subject to a Creative Commons CC-BY-NC-ND license.
Subject to such license, all rights are reserved.
The MIT Press would like to thank the anonymous peer reviewers who provided comments on drafts of this
book. The generous work of academic experts is essential for establishing the authority and quality of our
publications. We acknowledge with gratitude the contributions of these otherwise uncredited readers.
Printed and bound in the United States of America.
Library of Congress Cataloging-in-Publication Data
Names: Murphy, Kevin P., author.
Title: Probabilistic machine learning : advanced topics / Kevin P. Murphy.
Description: Cambridge, Massachusetts : The MIT Press, [2023] | Series:
Adaptive computation and machine learning series | Includes
bibliographical references and index.
Identifiers: LCCN 2022045222 (print) | LCCN 2022045223 (ebook) | ISBN
9780262048439 (hardcover) | ISBN 9780262376006 (epub) | ISBN
9780262375993 (pdf)
Subjects: LCSH: Machine learning. | Probabilities.
Classification: LCC Q325.5 .M873 2023 (print) | LCC Q325.5 (ebook) | DDC
006.3/1015192–dc23/eng20230111
LC record available at https://lccn.loc.gov/2022045222
LC ebook record available at https://lccn.loc.gov/2022045223
10 9 8 7 6 5 4 3 2 1

17
|              | Bayesian      |     | neural      |         | networks |     |
| ------------ | ------------- | --- | ----------- | ------- | -------- | --- |
| This chapter | is coauthored |     | with Andrew | Wilson. |          |     |
17.1 Introduction
Deep neural networks (DNNs) are usually trained using a (penalized) maximum likelihood objective
to find a single setting of parameters. However, large flexible models like neural networks can
represent many functions, corresponding to different parameter settings, which fit the training data
well, yet generalize in different ways. (This phenomenon is known as (see e.g.,
underspecification
[D’A+20]; see Figure 17.11 for an illustration.) Considering all of these different models together
can lead to improved accuracy and uncertainty representation. This can be done by computing the
| posterior | predictive | distribution | using | Bayesian | model averaging: |        |
| --------- | ---------- | ------------ | ----- | -------- | ---------------- | ------ |
| p(y x,    | )=         | p(y x,θ)p(θ  | )dθ   |          |                  | (17.1) |
| |         | D          | |            | |D    |          |                  |        |
(cid:90)
| where  |                 | θ).    |     |     |     |     |
| ------ | --------------- | ------ | --- | --- | --- | --- |
| p(θ    | )               | p(θ)p( |     |     |     |     |
| ma|iDn | ch∝allengesDin| |        |     |     |     |     |
The applying Bayesian inference to DNNs are specifying suitable priors, and
efficiently computing the posterior, which is challenging due to the large number of parameters and
the large datasets. The application of Bayesian inference to DNNs is sometimes called
Bayesian
deep learning or BDL. By contrast, the term deep Bayesian learning or DBL refers to the
use of deep models to help speed up Bayesian inference of “classical” models, usually by training
amortized inference networks that can be used as part of a variational inference or importance
sampling algorithm, as discussed in Section 10.1.5.) For more details on the topic of BDL, see e.g.,
| [PS17; Wil20; | WI20;  | Jos+22;  | Kha20; | Arb+23]. |     |     |
| ------------- | ------ | -------- | ------ | -------- | --- | --- |
| 17.2          | Priors | for BNNs |        |          |     |     |
To perform Bayesian inference for the parameters of a DNN, we need to specify a prior p(θ). [Nal18;
WI20; For22] discusses the issue of prior selection at length. Here we just give a brief summary of
| common | approaches. |     |     |     |     |     |
| ------ | ----------- | --- | --- | --- | --- | --- |

648
|        |          |     |        |     |     |     |     | Chapter | 17. Bayesian | neural | networks |
| ------ | -------- | --- | ------ | --- | --- | --- | --- | ------- | ------------ | ------ | -------- |
| 17.2.1 | Gaussian |     | priors |     |     |     |     |         |              |        |          |
Consider an MLP with one hidden layer with activation function φ and a linear output:
| f(x;θ)=W |     | φ(W | x+b | )+b |     |     |     |     |     |     | (17.2) |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ |
|          |     | 2   | 1   | 1   | 2   |     |     |     |     |     |        |
(If the output is nonlinear, such as a softmax transform, we can fold it into the loss function during
| training.)  | If  | we have | two    | hidden | layers | this     | becomes |         |     |     |        |
| ----------- | --- | ------- | ------ | ------ | ------ | -------- | ------- | ------- | --- | --- | ------ |
| f(x;θ)=W    |     |         | (φ(W   | φ(W    | x+b    | )+b ))+b |         |         |     |     | (17.3) |
|             |     | 3       |        | 2 1    | 1      | 2        | 3       |         |     |     |        |
| In general, |     | with    | hidden | layers | and    | a linear | output, | we have |     |     |        |
|             |     | L       | 1      |        |        |          |         |         |     |     |        |
−
(17.4)
| f(x;θ)=W |     | L   | ( φ(W | 1 x+b | 1 ))+b | L   |     |     |     |     |     |
| -------- | --- | --- | ----- | ----- | ------ | --- | --- | --- | --- | --- | --- |
···
We need to specify the priors for and for =1:L. The most common choice is to use a
|          |          |          |        |         | W   |     | b l |     |     |     |        |
| -------- | -------- | -------- | ------ | ------- | --- | --- | --- | --- | --- | --- | ------ |
|          |          |          |        |         |     | l   | l   |     |     |     |        |
| factored | Gaussian |          | prior: |         |     |     |     |     |     |     |        |
|          |          | (0,α2I), |        | (0,β2I) |     |     |     |     |     |     | (17.5) |
| W        | ℓ        |          | b ℓ    |         |     |     |     |     |     |     |        |
|          | ∼N       | ℓ        |        | ∼N      | ℓ   |     |     |     |     |     |        |
The or initialization, named after the first author of [GB10], is to
|     | Xavier | initialization |     |     | Glorot |     |     |     |     |     |     |
| --- | ------ | -------------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
set
|     |     | 2   |     |     |     |     |     |     |     |     | (17.6) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ |
α2 =
| ℓ   | n   | +n  |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
in out
where n is the fan-in of a node in level ℓ (number of weights coming into a neuron), and n is
|     | in  |     |     |     |     |     |     |     |     |     | out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the fan-out (number of weights going out of a neuron). LeCun initialization, named after Yann
| LeCun, | corresponds |     | to  | using |     |     |     |     |     |     |        |
| ------ | ----------- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | ------ |
|        | 1           |     |     |       |     |     |     |     |     |     | (17.7) |
α2 =
ℓ n
in
We can get a better understanding of these priors by considering the effect they have on the
corresponding distribution over functions that they define. To help understand this correspondence,
| let us | reparameterize |       | the  | model    | as follows: |        |       |     |     |     |        |
| ------ | -------------- | ----- | ---- | -------- | ----------- | ------ | ----- | --- | --- | --- | ------ |
| W      | =α             | η , η |      | (0,I), b | =β          | ϵ , ϵ  | (0,I) |     |     |     | (17.8) |
|        | ℓ ℓ            | ℓ     | ℓ ∼N |          | ℓ ℓ         | ℓ ℓ ∼N |       |     |     |     |        |
Hence every setting of the prior hyperparameters specifies the following random function:
(17.9)
| f(x;α,β)=α |     |     | L η L ( | φ(α 1 η | 1 x+β | 1 ϵ 1 ))+β | L ϵ L |     |     |     |     |
| ---------- | --- | --- | ------- | ------- | ----- | ---------- | ----- | --- | --- | --- | --- |
···
To get a feeling for the effect of these hyperparameters, we can sample MLP parameters from this
prior and plot the resulting random functions. We use a sigmoid nonlinearity, so φ(a)=σ(a). We
consider layers, so are the input-to-hidden weights, and are the hidden-to-output
|     | L   | = 2 |     | W 1 |     |     |     |     | W 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
weights. We assume the input and output are scalars, so we are generating random nonlinear 1d
| mappings | f   | :R  | .   |     |     |     |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
R
Figure 17.1(a→) shows some sampled functions where 5, 1, 1, 1. In
|     |     |     |     |     |     |     |     | α 1 = | β 1 = α 2 = | β 2 | =   |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ----------- | --- | --- |
Figure17.1(b)weincreaseα ; thisallowsthefirstlayerweightstogetbigger,makingthesigmoid-like
1
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

17.2. Priors for BNNs 649
10
0
10
−
1.0 0.5 0.0 0.5 1.0
− −
x
y
α =5, β =1, α =1, β =1
1 1 2 2
10
0
10
−
1.0 0.5 0.0 0.5 1.0
− −
x
(a)
y
α =25, β =1, α =1, β =1
1 1 2 2
(b)
10
0
10
−
1.0 0.5 0.0 0.5 1.0
− −
x
y
α =5, β =5, α =1, β =1
1 1 2 2
10
0
10
−
1.0 0.5 0.0 0.5 1.0
− −
x
(c)
y
α =5, β =1, α =5, β =1
1 1 2 2
(d)
Figure 17.1: The effects of changing the hyperparameters on an MLP with one hidden layer. (a) Random
functionssampledfromaGaussianpriorwithhyperparametersα =5,β =1,α =1,β =1. (b)Increasing
1 1 2 2
α by a factor of 5. (c) Increasing β by a factor of 5. (d) Increasing α by a factor of 5. Generated by
1 1 2
mlp_priors_demo.ipynb.
shape of the functions steeper. In Figure 17.1(c), we increase β ; this allows the first layer biases to
1
get bigger, which allows the center of the sigmoid to shift left and right more, away from the origin.
In Figure 17.1(d), we increase α ; this allows the second layer linear weights to get bigger, making
2
the functions more “wiggly” (greater sensitivity to change in the input, and hence larger dynamic
range).
The above results are specific to the case of sigmoidal activation functions. ReLU units can behave
differently. For example, [WI20, App. E] show that for MLPs with ReLU units, if we set β =0, so
ℓ
the bias terms are all zero, the effect of changing α is just to rescale the output. To see this, note
ℓ
that Equation (17.9) simplifies to
f(x;α,β =0)=α η ( φ(α η x))=α α η ( φ(η x)) (17.10)
L L ··· 1 1 L ··· 1 L ··· 1
=α α f(x;(α=1,β =0)) (17.11)
L 1
···
where we used the fact that for ReLU, φ(αz) = αφ(z) for any positive α, and φ(αz) = 0 for any
negative α (since the preactivation z 0). In general, it is the ratio of α and β that matters for
determining what happens to input si≥gnals as they propagate forwards and backwards through a
randomly initialized model; for details, see e.g., [Bah+20].
We see that initializing the model’s parameters at a particular random value is like sampling a
Author: Kevin P. Murphy. (C) MIT Press. CC-BY-NC-ND license

650
|     |     |     | Chapter 17. | Bayesian neural | networks |
| --- | --- | --- | ----------- | --------------- | -------- |
point from this prior over functions. In the limit of infinitely wide neural networks, we can derive
this prior distribution analytically: this is known as a process, and is
|                           |               | neural | network | Gaussian |     |
| ------------------------- | ------------- | ------ | ------- | -------- | --- |
| explained in              | Section 18.7. |        |         |          |     |
| 17.2.2 Sparsity-promoting |               | priors |         |          |     |
Although Gaussian priors are simple and widely used, they are not the only option. For some
applications, it is useful to use priors, such as the Laplace, which encourage
sparsity promoting
most of the weights (or channels in a CNN) to be zero (cf. Section 15.2.6). For details, see [Hoe+21].
| 17.2.3 Learning | the prior |     |     |     |     |
| --------------- | --------- | --- | --- | --- | --- |
We have seen how different priors for the parameters correspond to different priors over functions.
We could in principle set the hyperparameters (e.g., the and parameters of the Gaussian prior)
α β
using grid search to optimize cross-validation loss. However, cross-validation can be slow, particularly
if we allow different priors for each layer of the network, as our grid search will grow exponentially
| with the number | of hyperparameters | we wish to determine. |     |     |     |
| --------------- | ------------------ | --------------------- | --- | --- | --- |
An alternative is to use gradient based methods to optimize the marginal likelihood
| logp( α,β)= | logp( | θ)p(θ α,β)dθ |     |     | (17.12) |
| ----------- | ----- | ------------ | --- | --- | ------- |
| D|          | D|    | |            |     |     |         |
(cid:90)
This approach is known as empirical Bayes (Section 3.7) or evidence maximization, since
is also called the evidence [Mac92a; WS93; Mac99]. This can give rise to sparse
logp( α,β)
modelDs,|as
we discussed in the context of automatic relevancy determination (Section 15.2.8). Unfor-
tunately, computing the marginal likelihood is computationally difficult for large neural networks.
Learning the prior is more meaningful if we can do it on a separate, but related dataset. In
[SZ+22] they propose to train a model on an initial, large dataset (possibly unsupervised) to
1
get a point estimate, θˆ , from which they can derive an approximate D low-rank Gaussian posterior,
1
using the SWAG method (Section 17.3.8). They then use this informative prior when fine-tuning
the model on a downstream dataset . The fine-tuning can either be a MAP estimate θˆ or some
|     |     | ),De.g., 2 |     |     | 2   |
| --- | --- | ---------- | --- | --- | --- |
approximate posterior, p(θ , computed using MCMC (Section 17.3.7). They call this
2 2 1
technique “Bayesian | D D arning”. (See Section 19.5.1 for more details on transfer learning.)
|               | transf e    | r le  |     |     |     |
| ------------- | ----------- | ----- | --- | --- | --- |
| 17.2.4 Priors | in function | space |     |     |     |
Typically, the relationship between the prior distribution over parameters and the functions preferred
by the prior is not transparent. In some cases, it can be possible to pick more informative priors
based on principles such as desired invariances that we want the function to satisfy (see e.g., [Nal18]).
[FBW21] introduces priors, providing a mechanism for encoding high level concepts
|     | residual | pathway |     |     |     |
| --- | -------- | ------- | --- | --- | --- |
into prior distributions, such as locality, independencies, and symmetries, without constraining model
flexibility. A different approach to encoding interpretable priors over functions leverages kernel
methods such as Gaussian processes (e.g., [Sun+19a]), as we discuss in Section 18.1.
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

651
| 17.3. Posteriors | for BNNs      |        |     |     |     |     |     |
| ---------------- | ------------- | ------ | --- | --- | --- | --- | --- |
| 17.2.5           | Architectural | priors |     |     |     |     |     |
Beyond specifying the parametric prior, it is important to note that the architecture of the model
can have an even larger effect on the induced distribution over functions, as argued in Wilson and
Izmailov [WI20] and Izmailov et al. [Izm+21b]. For example, a CNN architecture encodes prior
knowledge about translation equivariance, due to its use of convolution, and hierarchical structure,
due to its use of multiple layers. Other forms of inductive bias are induced by different architectures,
such as RNNs. (Models such as transformers have weaker inductive bias, but consequently often
need more data to perform well.) Thus we can think of the field of neural architecture search
| (reviewed | in [EMH19]) | as a form of structural | prior | learning. |     |     |     |
| --------- | ----------- | ----------------------- | ----- | --------- | --- | --- | --- |
Infact,withasuitablearchitecture,wecanoftengetgoodresultsusingrandom(untrained)models.
Forexample, Ulyanov, Vedaldi, andLempitsky[UVL18]showedthatanuntrainedCNNwithrandom
parameters (sampled from a Gaussian) often works very well for low-level image processing tasks,
such as image denoising, super-resolution, and image inpainting. The resulting prior over functions
has been called the prior. Similarly, Pinto and Cox [PC12] showed that untrained
|     | deep | image |     |     |     |     |     |
| --- | ---- | ----- | --- | --- | --- | --- | --- |
CNNs with the right structure can do well at face recognition. Moreover, Zhang et al. [Zha+17]
show that randomly initialized CNNs can process data to provide features that greatly improve the
| performance | of other   | models, such as kernel | methods. |     |     |     |     |
| ----------- | ---------- | ---------------------- | -------- | --- | --- | --- | --- |
| 17.3        | Posteriors | for BNNs               |          |     |     |     |     |
There are a large number of different approximate inference schemes that have been applied to
Bayesian neural networks, with different strengths and limitations. In the sections below, we briefly
| describe | some of these. |              |          |               |            |             |     |
| -------- | -------------- | ------------ | -------- | ------------- | ---------- | ----------- | --- |
| 17.3.1   | Monte Carlo    | dropout      |          |               |            |             |     |
|          |                | (MCD) [GG16; | KG17] is | a very simple | and widely | used method | for |
| Monte    | Carlo dropout  |              |          |               |            |             |     |
approximating the Bayesian predictive distribution. Usually stochastic dropout layers are added as a
form of regularization, and are “turned off” at test time, as described in Section 16.2.6, However, the
idea in MCD is to also perform random sampling at test time. More precisely, we drop out each
hidden unit by sampling from a Bernoulli(p) distribution; we repeat this procedure S times, to create
distinct models. We then create an equally weighted average of the predictive distributions for
S
| each of these | models: |     |     |     |     |     |     |
| ------------- | ------- | --- | --- | --- | --- | --- | --- |
1 S
|        |       | x,θs) |     |     |     |     | (17.13) |
| ------ | ----- | ----- | --- | --- | --- | --- | ------- |
| p(y x, | ) p(y |       |     |     |     |     |         |
| |      | D ≈ S | |     |     |     |     |     |         |
s=1
(cid:88)
where θs is a version of the MAP parameter estimate where we randomly drop out some connections.
We give an example of this process in action in Figure 17.2. We see that it succesfully captures
uncertainty due to “out of distribution” inputs. (See Section 19.3.2 for more discussion of OOD
detection.)
One drawback of MCD is that it is slow at test time. However this can be overcome by “distilling”
the model’s predictions into a deterministic “student” network, as we discuss in Section 17.3.10.3.
AmorefundamentalproblemisthatMCDdoesnotgiveproperuncertaintyestimates,asarguedin
[Osb16; LF+21]. Theproblemisthefollowing. AlthoughMCDcanbeviewedasaformofvariational
| Author: | Kevin P. Murphy. | (C) MIT Press. | CC-BY-NC-ND | license |     |     |     |
| ------- | ---------------- | -------------- | ----------- | ------- | --- | --- | --- |

652
|     |     |     |     |     |     |     |     | Chapter |     | 17. Bayesian | neural networks |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | ------------ | --------------- |
1.0
|     |     | 20  |     | 1   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
5
0.8
7
|     |        | 10  |     |     |     | ytilibaborP |     |     |     |     |     |
| --- | ------ | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
|     | stigoL |     |     |     |     |             | 0.6 |     |     |     |     |
1
|     |     |     |     |     |     |     | 0.4 |     | 5   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0
7
0.2
10
|     | −   |     |         |           |       |     | 0.0 |     |     |         |            |
| --- | --- | --- | ------- | --------- | ----- | --- | --- | --- | --- | ------- | ---------- |
|     |     |     | 1 2 3 4 | 5 6 7 8 9 | 10 11 | 12  |     | 1 2 | 3 4 | 5 6 7 8 | 9 10 11 12 |
|     |     |     |         | (a)       |       |     |     |     |     | (b)     |            |
Figure 17.2: Illustration of MC dropout applied to the LeNet architecture. The inputs are some rotated images
of the digit 1 from the MNIST dataset. (a) Softmax inputs (logits). (b) Softmax outputs (proabilities). We see
that the inputs are classified as digit 7 for the last three images (as shown by the probabilities), even though
the model has high uncertainty (as shown by the logits). Adapted from Figure 4 of [GG16]. Generated by
mnist_classification_mc_dropout.ipynb
inference [GG16], this is only true under a degenerate posterior approximation, corresponding to a
mixture of two delta functions, one at 0 (for dropped out nodes) and one at the MLE. This posterior
will not converge to the true posterior (which is a delta function at the MLE) even as the training
set size goes to infinity, since we are always dropping out hidden nodes with a constant probability
p
[Osb16]. Fortunately this pathology can be fixed if the noise rate is optimized [GHK17]. For more
| details, | see | e.g.,   | [HGMG18;      | NHLS19; | LF+21]. |     |     |     |     |     |     |
| -------- | --- | ------- | ------------- | ------- | ------- | --- | --- | --- | --- | --- | --- |
| 17.3.2   |     | Laplace | approximation |         |         |     |     |     |     |     |     |
InSection7.4.3,weintroducedtheLaplaceapproximation,whichcomputesaGaussianapproximation
to the posterior, ), centered at the MAP estimate, . The posterior prediction matrix is equal
|     |     |     | p(θ           |     |     |     | θ   |     |     |     |     |
| --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | th|eDnegative |     |     |     |     | ∗   |     |     |     |
to the Hessian of log joint computed at the mode. The benefits of this approach are
that it is simple, and it can be used to derive a Bayesian estimate from a pretrained model. The
main disadvantage is that computing the Hessian can be expensive. In addition, it may not be
positive definite, since the log likelihood of DNNs is non-convex. It is therefore common to use a
Gauss-newton approximation to the Hessian instead, as we explain below.
Following the notation of [Dax+21], let f(x ,θ) C be the prediction function with C outputs,
|     |     |      |               |         |         | n          | R   |        |     |                |         |
| --- | --- | ---- | ------------- | ------- | ------- | ---------- | --- | ------ | --- | -------------- | ------- |
| and |     | P be | the parameter | vector. | Let     |            | ∈   |        | be  | the residual1, | and     |
|     | θ   | R    |               |         | r(y;f)= |            | f   | logp(y | f)  |                | Λ(y;f)= |
|     | ∈   |      |               |         |         | additi∇on, |     |        | |   |                |         |
2 lo gp(y f) be the per-input noise term. In let J R C P be the Jacobian, [J (x)] =
| −∇  | f   | |   |     |     |     |     |     | ∈   | ×   |     | θ ci |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
∂2
∂fc ( x ,θ), and H C P P be the Hessian, [H (x)] = f c ( x , θ). Then the gradient and Hessian
|     | ∂ θ |     | R × | ×   |     | θ     | cij | ∂ θ ∂ | θ   |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | ----- | --- | --- | --- |
|     | i   |     | ∈   |     |     |       |     | i     | j   |     |     |
|     |     |     |     |     | y   | f 2=2 | y f |       |     |     |     |
1. IntheGaussiancase,thistermbecomes f ,soitcanbeinterpretedasaresidualerror.
|     |     |     |     |     | ∇ || − | ||  | || − | ||  |     |     |     |
| --- | --- | --- | --- | --- | ------ | --- | ---- | --- | --- | --- | --- |
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

653
| 17.3. | Posteriors |            | for BNNs |            |        |           |             |       |     |     |         |
| ----- | ---------- | ---------- | -------- | ---------- | ------ | --------- | ----------- | ----- | --- | --- | ------- |
| of    | the log    | likelihood | are      | given      | by the | following | [IKB21]:    |       |     |     |         |
|       | logp(y     | f(x,θ))=J  |          | (x)Tr(y;f) |        |           |             |       |     |     | (17.14) |
|       | θ          |            |          | θ          |        |           |             |       |     |     |         |
|       | ∇          | |          |          |            |        |           |             |       |     |     |         |
|       | 2logp(y    |            |          | (x)Tr(y;f) |        |           | (x)TΛ(y;f)J |       |     |     | (17.15) |
|       |            | f(x,θ))=H  |          | θ          |        | J         | θ           | θ (θ) |     |     |         |
|       | ∇θ         | |          |          |            |        | −         |             |       |     |     |         |
Since the network Hessian H is usually intractable to compute, it is usually dropped, leaving only
the Jacobian term. This is called the or approximation [Sch02;
|     |     |     |     |     |     | generalized | Gauss-Newton |     | GGN |     |     |
| --- | --- | --- | --- | --- | --- | ----------- | ------------ | --- | --- | --- | --- |
Mar20]. The GGN approximation is guaranteed to be positive definite. By contrast, this is not
true for the original Hessian in Equation (17.15), since the objective is not convex. Furthermore,
| computing |     | the Jacobian |     | term | is cheaper | to  | compute | than the Hessian. |     |     |     |
| --------- | --- | ------------ | --- | ---- | ---------- | --- | ------- | ----------------- | --- | --- | --- |
Putting it all together, for a Gaussian prior, p(θ) = (θ m ,S ), the Laplace approximation
|         |     |       |      |          |          |     |     | 0   | 0   |     |     |
| ------- | --- | ----- | ---- | -------- | -------- | --- | --- | --- | --- | --- | --- |
| becomes |     |       |      |          | ), where |     |     | N | |     |     |     |
|         |     | p(θ ) | (    | θ ,Σ GGN |          |     |     |     |     |     |     |
|         |     | |D    | ≈ N| | ∗        |          |     |     |     |     |     |     |
N
|     | 1    |     |         | )TΛ(y |           |               | 1   |     |     |     | (17.16) |
| --- | ---- | --- | ------- | ----- | --------- | ------------- | --- | --- | --- | --- | ------- |
|     | Σ−GG | = J | θ∗ (x n |       | n ;f n )J | θ∗ (x n )+S−0 |     |     |     |     |         |
N
n=1
(cid:88)
Unfortunately inverting this matrix takes O(P3) time, so for models with many parameters, further
approximations are usually used. The simplest is to use a diagonal approximation, which takes
O(P)
timeandspace. Amoresophisticatedapproachispresentedin[RBB18a], whichleveragestheKFAC
(Kronecker factored curvature) approximation of [MG15]. This approximates the covariance of each
| layer | using | a Kronecker |     | product. |     |     |     |     |     |     |     |
| ----- | ----- | ----------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
A limitation of the Laplace approximation is that the posterior covariance is derived from the
Hessian evaluated at the MAP parameters. This means Laplace forms a highly approximation:
local
even if the non-Gaussian posterior could be well-described by a Gaussian distribution, the Gaussian
distribution formed using Laplace only captures the local characteristics of the posterior at the
MAP parameters — and may therefore suffer badly from local optima, providing overly compact
or diffuse representations. In addition, the curvature information is only used after the model has
been estimated, and not during the model optimization process. By contrast, variational inference
(Section 17.3.3) can provide more accurate approximations for comparable cost.
| 17.3.3 |     | Variational |     | inference |     |     |     |     |     |     |     |
| ------ | --- | ----------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
In fixed-form variational inference (Section 10.2), we choose a distribution for the posterior approxi-
mation q (θ)and minimize DKL(q p), with respect to ψ. We often choose a Gaussian approximate
ψ
posterior, µ,Σ), whic∥h lets us use the reparameterization trick to create a low variance
|     |     | q (θ)= | (θ         |     |     |     |     |     |     |     |     |
| --- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | ψ      | grNadie|nt |     |     |     |     |     |     |     |     |
estimator of the of the ELBO (see Section 10.2.1). Despite the use of a Gaussian, the
parameters that minimize the KL objective are often different from what we would find with the
| Laplace |     | approximation |     | (Section | 17.3.2). |     |     |     |     |     |     |
| ------- | --- | ------------- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- |
Variational methods for neural networks date back to at least Hinton and Camp [HC93]. In deep
learning, [Gra11] revisited variational methods, using a Gaussian approximation with a diagonal
covariancematrix. Thisapproximatesthedistributionofeveryparameterinthemodelbyaunivariate
Gaussian,wherethemeanisthepointestimate,andthevariancecapturestheuncertainty,asshownin
Figure 17.3. This approach was improved further in [Blu+15], who used the reparameterization trick
to compute lower variance estimates of the ELBO; they called their method Bayes by backprop
(BBB). This is essentially identical to the SVI algorithm in Algorithm 10.2, except the likelihood
becomesp(y x ,θ)fromtheDNN,andthepriorp (θ)andvariationalposteriorq (θ)areGaussians.
|     |     | n n |     |     |     |     | ξ   |     |     | ψ   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|
| Author: |     | Kevin P. | Murphy. | (C) | MIT | Press. | CC-BY-NC-ND | license |     |     |     |
| ------- | --- | -------- | ------- | --- | --- | ------ | ----------- | ------- | --- | --- | --- |

654
|     |     |     |     |     | Chapter 17. | Bayesian neural | networks |
| --- | --- | --- | --- | --- | ----------- | --------------- | -------- |
|     |     | h   |     |     |             | h               |          |
|     |     | 4   |     |     |             | 4               |          |
0.1
0 .2 5
| x 2 | −0 .2 5 | h   | 1.25 |     | x 2 | h   |     |
| --- | ------- | --- | ---- | --- | --- | --- | --- |
|     |         | 3   |      |     |     | 3   |     |
|     |         |     | 0.9  | y   |     |     | y   |
0.4
|     | 0.1  |     | 0.55 |     |     |     |     |
| --- | ---- | --- | ---- | --- | --- | --- | --- |
| x − |      | h   |      |     | x   | h   |     |
| 1   | 0.05 | 2   |      |     | 1   | 2   |     |
0.55
0.2
0.2
|     |     | h   |     |     |     | h   |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | 1   |     |     |     | 1   |     |
Figure17.3: IllustrationofanMLPwith(left)apointestimateforeachweight,(right)amarginaldistribution
| for each weight, | corresponding | to  | a fully | factored posterior | approximation. |     |     |
| ---------------- | ------------- | --- | ------- | ------------------ | -------------- | --- | --- |
Many extensions of the BBB have been proposed. In [KSW15], they propose the local repa-
trick, that samples the activations at each layer, instead of the weights
| rameterization |     |     |     |     | a=Wz |     |     |
| -------------- | --- | --- | --- | --- | ---- | --- | --- |
W, which results in a lower variance estimate of the ELBO gradient. In [Osa+19a], they used
the variational online Gauss-Newton (VOGN) method of [Kha+18], for improved scalability.
VOGN is a noisy version of natural gradient descent, where the extra noise emulates the effect
of variational inference. In [Mis+18], they replaced the diagonal approximation with a low-rank
plus diagonal approximation, and used VOGN for fitting. In [Tra+20b], they use a rank-one plus
diagonal approximation known as NAGVAC (see Section 10.2.1.3). In this case, there are only 3
times as many parameters as when computing a point estimate (for the variational mean, variance,
and rank-one vector), making the approach very scalable. In addition, in this case it is possible to
analytically compute the natural gradient, which speeds up model fitting (see Section 6.4). Many
other variational methods have also been proposed (see e.g., [LW16; Zha+18; Wu+19a; HHK19]).
| See also Section   | 17.5.4 | for a discussion |     | of online VI | for DNNs. |     |     |
| ------------------ | ------ | ---------------- | --- | ------------ | --------- | --- | --- |
| 17.3.4 Expectation |        | propagation      |     |              |           |     |     |
Expectationpropagation(EP)issimilartovariationalinference,exceptitlocallyoptimizesDKL(p
q)
deta∥ils,
instead of DKL(q p), where p is the exact posterior and q is the approximate posterior. For
∥
| see Section | 10.7. |     |     |     |     |     |     |
| ----------- | ----- | --- | --- | --- | --- | --- | --- |
A special case of EP is the assumed density filtering (ADF) algorithm of Section 8.6, which is
equivalent to the first pass of ADF. In Section 8.6.3 we show how to apply ADF to online logistic
regression. In[HLA15a],theyextendADFtothecaseofBNNs;theycalledtheirmethodprobabilistic
backpropagation or PBP. They approximate every parameter in the model by a Gaussian factor, as
| in Figure 17.3. | See   | Section 17.5.3 | for | the details. |     |     |     |
| --------------- | ----- | -------------- | --- | ------------ | --- | --- | --- |
| 17.3.5 Last     | layer | methods        |     |              |     |     |     |
A very simple approximation to the posterior is to only “be Bayesian” about the weights in the final
layer, and to use MAP estimates for all the other parameters. This is called the neural-linear
approximation [RTS18]. In more detail, let be the predicted outputs (e.g., logits) of
z = f(x,θ)
the model before any optional final nonlinearity. We assume this has the form z = wTϕ(x;θ),
L
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

655
| 17.3. | Posteriors |     | for BNNs |     |     |     |     |
| ----- | ---------- | --- | -------- | --- | --- | --- | --- |
where are the features extracted by the first layers. This gives us a Bayesian GLM.
|     | ϕ(x) |     |     |     |     |     | L 1 |
| --- | ---- | --- | --- | --- | --- | --- | --- |
We can use standard techniques, such as the Laplace a−pproximation (Section 15.3.5), to compute
p(w ) = (µ ,Σ ), given ϕ(). To estimate the parameters of the feature extractor, we can
|     | L   |     | L L |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
optim |iDze theNlog-l ikeli hood in the usual way. Given the posterior over the last layer weights, we can
| compute |     | the posterior | predictive |     | distribution | over the | logits using |
| ------- | --- | ------------- | ---------- | --- | ------------ | -------- | ------------ |
ϕ(x)T) (17.17)
|     | p(z x, | )=  | (z µ  | ϕ(x),ϕ(x)Σ |     |     |     |
| --- | ------ | --- | ----- | ---------- | --- | --- | --- |
|     | |      | D   | N | L |            | L   |     |     |
Thiscanbepassedthroughthefinalsoftmaxlayertocomputep(y x, )asdescribedinSection15.3.6.
In[KHH20]theyshowthiscanreduceoverconfidenceinpredict|ionsDforinputsthatarefarfromthe
trainingdata. However,thisapproachignoresuncertaintyintroducedbytheearlierfeatureextraction
layers, where most of the parameters reside. We discuss a solution to this in Section 17.3.6.
| 17.3.6 |     | SNGP |     |     |     |     |     |
| ------ | --- | ---- | --- | --- | --- | --- | --- |
It is possible to combine DNNs with Gaussian process (GP) models (Chapter 18), by using the DNN
to act as a feature extractor, which is then fed into the kernel in the final layer. This is called “deep
| kernel | learning” |     | (see Section | 18.6.6). |     |     |     |
| ------ | --------- | --- | ------------ | -------- | --- | --- | --- |
One problem with this is that the feature extractor may lose information which is not needed for
classification accuracy, but which is needed for robust performance on out-of-distribution inputs (see
Section 17.4.6.2). The basic problem is that, in a classification problem, there is no reduction in
training accuracy (log likelihood) if points which are far away are projected close together, as long as
they are on the correct side of the decision boundary. Thus the distances between two inputs can be
erased by the feature extraction layers, so that OOD inputs appear to the final layer to be close to
| the | training | set. |     |     |     |     |     |
| --- | -------- | ---- | --- | --- | --- | --- | --- |
One solution to this is to use the SNGP (spectrally normalized Gaussian process) method of
[Liu+20d; Liu+22a]. This constrains the feature extraction layers to be “distance preserving”, so
that two inputs that are far apart in input space remain far apart after many layers of feature
extraction, by using spectral normalization of the weights to bound the Lipschitz constant of the
feature extractor. The overall approach ensures that information that is relevant for computing the
confidence of a prediction, but which might be irrelevant to computing the label of a prediction, is
not lost. This can help performance in tasks such as out-of-distribution detection (Section 17.4.6.2).
| 17.3.7 |     | MCMC | methods |     |     |     |     |
| ------ | --- | ---- | ------- | --- | --- | --- | --- |
Some of the earliest work on inference for BNNs was done by Radford Neal, who proposed to use
Hamiltonian Monte Carlo (Section 12.5) to approximate the posterior [Nea96]. This is generally
considered the gold standard method, since it does not make strong assumptions about the form of
the posterior. For more recent work on scaling up HMC for BNNs, see e.g., [Izm+21b; CJ21].
We give a simple example of vanilla HMC in Figure 17.4, where we fit a shallow MLP to a small
2d binary dataset. We plot the mean and standard deviation of the posterior predictive distribution,
). We see that the uncertainty is higher as we move away from the training data.
p(y = 1x;
| (Compar|e |     | toDBayesian |          |            |     |                 |         |
| --------- | --- | ----------- | -------- | ---------- | --- | --------------- | ------- |
|           |     |             | logistic | regression |     | in 1d in Figure | 15.8a.) |
However, a significant limitation of standard MCMC procedures, including HMC, is that they
require access to the full training set at each step. Stochastic gradient MCMC methods, such as
| Author: |     | Kevin | P. Murphy. | (C) | MIT Press. | CC-BY-NC-ND | license |
| ------- | --- | ----- | ---------- | --- | ---------- | ----------- | ------- |

656
|     |     |               |     |     | Chapter | 17.          | Bayesian neural | networks |
| --- | --- | ------------- | --- | --- | ------- | ------------ | --------------- | -------- |
|     |     | Posteriormean |     |     |         | Posteriorstd |                 |          |
0.54
0.48
0.42
0.36
0.30
0.24
0.18
0.12
0.06
0.00
|     |     | (a) |     |     |     |     | (b) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
Figure 17.4: Illustration of an MLP fit to the two-moons dataset using HMC. (a) Posterior mean. (b)
Posterior standard derivation. The uncertainty increases as we move away from the training data. Generated
by bnn_mlp_2d_hmc.ipynb.
SGLD, operate instead using mini-batches of data, offering a scalable alternative, as we discuss in
Section 12.7.1. For an example of SGLD applied to an MLP, see Section 19.3.3.1.
| 17.3.8 Methods |     | based | on the | SGD trajectory |     |     |     |     |
| -------------- | --- | ----- | ------ | -------------- | --- | --- | --- | --- |
In [MHB17; SL18; CS18], it was shown that, under some assumptions, the iterates produced by
stochastic gradient descent (SGD), when run at a fixed learning rate, correspond to samples from
a Gaussian approximation to the posterior centered at a local mode, θˆ,Σ). We can
|     |     |     |     |     |     |     | p(θ ) (θ |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- |
Thisissimil|aDrto≈SNG-M|CMCmethods,
thereforeuseSGDtogenerateapproximateposteriorsamples.
except we do not add explicit gradient noise, and the learning rate is held constant.
In[Izm+18], theynotedthattheseSGDsolutions(withfixedlearningrate)surroundtheperiphery
of points of good generalization, as shown in Figure 17.5. This is in part because SGD does not
converge to a local optimum unless the learning rate is annealed to 0. They therefore proposed to
compute the average of several SGD samples, each one collected after a certain interval (e.g., one
| epoch of training), | to  | get | 1 S   | . They call | this       |     |                  | (SWA). |
| ------------------- | --- | --- | ----- | ----------- | ---------- | --- | ---------------- | ------ |
|                     |     | θ = |       | θ           | stochastic |     | weight averaging |        |
|                     |     |     | S s=1 | s           |            |     |                  |        |
They showed that the resulting point tends to correspond to a broader local minimum than the SGD
solutions (see Figure 17.10), resul(cid:80)ting in better generalization performance.
The SWA approach is related to Polyak-Ruppert averaging, which is often used in convex optimiza-
tion. The difference is that Polyak-Ruppert typically assumes the learning rate decays to zero, and
uses an exponential moving average (EMA) of iterates, rather than an equal average; Polyak-Ruppert
averaging is mainly used to reduce variance in the SGD estimate, rather than as a method to find
| points of better | generalization. |         |            |                    |     |            |      |           |
| ---------------- | --------------- | ------- | ---------- | ------------------ | --- | ---------- | ---- | --------- |
| The SWA approach |                 | is also | related to |                    |     | [Hua+17a], | and  |           |
|                  |                 |         |            | snapshot ensembles |     |            | fast | geometric |
ensembles [Gar+18c]; these methods save the parameters θ after increasing and decreasing the
s
learning rate multiple times in a cyclical fashion, and then computing the
|     |     |     |     |     |     |     | average of the | predictions |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ----------- |
using p(y x, ) 1 S p(y x,θ ), rather than computing the average of the parameters and
|     | S   | s=1 | s   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | D | ≈   |     | |   |     |     |     |     |     |
(cid:80)
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

657
| 17.3. | Posteriors | for | BNNs |     |     |     |     |     |     |     |
| ----- | ---------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
Figure 17.5: Illustration of stochastic weight averaging (SWA). The three crosses represent different SGD
solutions. The star in the middle is the average of these parameter values. From Figure 1 of [Izm+18]. Used
| with | kind permission |     | of Andrew Wilson. |     |     |     |     |     |     |     |
| ---- | --------------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
predicting with a single model (which is faster). Moreover, by finding a flat region, representing a
“center or mass” in the posterior, SWA can be seen as approximating the Bayesian model average in
| Equation | 17.1 | with | a single model. |     |     |     |     |     |     |     |
| -------- | ---- | ---- | --------------- | --- | --- | --- | --- | --- | --- | --- |
In [Mad+19], they proposed to fit a Gaussian distribution to the set of samples produced by SGD
near a local mode. They use the SWA solution as the mean of the Gaussian. For the covariance
matrix, they use a low-rank plus diagonal approximation of the form θ,Σ), where
|     |     |     |     |     |     |     |     | p(θ ) | = (θ |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ---- | --- |
|     |     |     |     |     |     |     |     | |D    | N |  |     |
Σ=(Σ +Σ )/2, Σ =diag(θ2 (θ)2), θ = 1 S θ , θ2 = 1 S θ2, and Σ = 1∆∆T
|     | diag | lr  | diag |     |     | S   | s=1 s | S s=1 s | lr  | S   |
| --- | ---- | --- | ---- | --- | --- | --- | ----- | ------- | --- | --- |
−
is the sample covariance matrix of the last K samples of ∆ = (θ θ ), where θ is the running
|     |     |     |     |     |     | (cid:80) | i   | i e−tho (cid:80) i | i   |     |
| --- | --- | --- | --- | --- | --- | -------- | --- | ------------------ | --- | --- |
average of the parameters from the first i samples. Th ey call this m d SWAG, w hich stands for
“stochastic weight averaging with Gaussian posterior”. This can be used to generate an arbitrary
number of posterior samples at prediction time. They show that SWAG scales to large residual
networks with millions of parameters, and large datasets such as ImageNet, with improved accuracy
and calibration over conventional SGD training, and no additional training overhead.
| 17.3.9 |     | Deep ensembles |     |     |     |     |     |     |     |     |
| ------ | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Many conventional approximate inference methods focus on approximating the posterior p(θ ) in a
local neighborhood around one of the posterior modes. While this is often not a major limit|aDtion in
classical machine learning, modern deep neural networks have highly multi-modal posteriors, with
parametersindifferentmodesgivingrisetoverydifferentfunctions. Ontheotherhand,thefunctions
in a neighborhood of a single mode may make fairly similar predictions. So using such a local
approximation to compute the posterior predictive will underestimate uncertainty and generalize
more poorly.
A simple alternative method is to train multiple models, and then to approximate the posterior
| using | an equally | weighted | mixture | of delta | functions, |     |     |     |     |     |
| ----- | ---------- | -------- | ------- | -------- | ---------- | --- | --- | --- | --- | --- |
M
1
| p(θ | )   |     | δ(θ θˆ ) |     |     |     |     |     |     | (17.18) |
| --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | ------- |
m
|     | |D  | ≈ M | −   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
m=1
(cid:88)
where M is the number of models, and θˆ is the MAP estimate for model m. See Figure 17.6 for a
m
| Author: | Kevin | P. Murphy. | (C) | MIT Press. | CC-BY-NC-ND |     | license |     |     |     |
| ------- | ----- | ---------- | --- | ---------- | ----------- | --- | ------- | --- | --- | --- |

658
|     |     |     |     |     | Chapter 17. | Bayesian neural | networks |
| --- | --- | --- | --- | --- | ----------- | --------------- | -------- |
Figure 17.6: Cartoon illustration of the NLL as it varies across the parameter space. Subspace methods
(red) model the local neighborhood around a local mode, whereas ensemble methods (blue) approximate the
posterior using a set of distinct modes. From Figure 1 of [FHL19]. Used with kind permission of Balaji
Lakshminarayanan.
| sketch. | This approach | is called deep | ensembles | [LPB17; | FHL19]. |     |     |
| ------- | ------------- | -------------- | --------- | ------- | ------- | --- | --- |
The models can differ in terms of their random seed used for initialization [LPB17], or hyper-
parameters [Wen+20c], or architecture [Zai+20], or all of the above. In addition, [DF21; TB22]
discusses how to add an explicit repulsive term to ensure functional diversity between the ensemble
members. This way, each member corresponds to a distinct prediction function. Combining these is
more effective than combining multiple samples from the same basin of attraction, especially in the
| presence | of dataset shift | [Ova+19]. |     |     |     |     |     |
| -------- | ---------------- | --------- | --- | --- | --- | --- | --- |
| 17.3.9.1 | Multi-SWAG       |           |     |     |     |     |     |
We can further improve on this approach by fitting a Gaussian to each local mode using the SWAG
method from Section 17.3.8 to get a mixture of Gaussians approximation:
1 M
|     |     | θˆ          |     |     |     |     | (17.19) |
| --- | --- | ----------- | --- | --- | --- | --- | ------- |
| p(θ | )   | (θ m ,Σ m ) |     |     |     |     |         |
| |D  | ≈ M | N |         |     |     |     |     |         |
m=1
(cid:88)
This approach is known as MultiSWAG [WI20]. MultiSWAG performs a Bayesian model average
bothacrossmultiplebasinsofattraction,likedeepensembles,butalsowithineachbasin,andprovides
an easy way to generate an arbitrary number of posterior samples, S >M, in an any-time fashion.
| 17.3.9.2 | Deep ensembles | with | random | priors |     |     |     |
| -------- | -------------- | ---- | ------ | ------ | --- | --- | --- |
The standard way to fit each member of a deep ensemble is to initialize them each with a different
random set of parameters, but them to train them all on the same data. Unfortunately this can
result in the predictions from each ensemble member being rather similar, which reduces the benefit
of the approach. One way to increase diversity is to train each member on a different subset of the
data; this is called bootstrap sampling. Another approach is to define the i’th ensemble member
to be the addition of a trainable model and a fixed, but random, network, (x),
| g (x) |     |     |     | t (x) |     | prior | p   |
| ----- | --- | --- | --- | ----- | --- | ----- | --- |
| i     |     |     |     | i     |     |       | i   |
to get
| g (x;θ | )=t (x;θ | )+βp (x) |     |     |     |     | (17.20) |
| ------ | -------- | -------- | --- | --- | --- | --- | ------- |
| i      | i i      | i i      |     |     |     |     |         |
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

659
| 17.3. | Posteriors | for BNNs |     |         |     |     |         |     |         |     |
| ----- | ---------- | -------- | --- | ------- | --- | --- | ------- | --- | ------- | --- |
|       | model#1    |          |     | model#2 |     | 2   |         |     |         |     |
| 1     |            |          |     |         |     |     | β=0.001 |     | β=0.001 |     |
0
0
2
| 1   | model#3 |     |     | model#4 |     |     | β=5 |     | β=5 |     |
| --- | ------- | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
0
0
2
|     | model#5 |     |     | model#6 |     |     |       |     |       |     |
| --- | ------- | --- | --- | ------- | --- | --- | ----- | --- | ----- | --- |
| 1   |         |     |     |         |     |     | β=50  |     | β=50  |     |
| 0   |         |     |     |         |     | 0   |       |     |       |     |
|     | model#7 |     |     | model#8 |     | 2   |       |     |       |     |
| 1   |         |     |     |         |     |     | β=100 |     | β=100 |     |
0
0
|     | -0.5 0 | 0.5 | 1 -0.5 | 0   | 0.5 | 1 -0.5 | 0   | 0.5 1 -0.5 | 0 0.5 | 1   |
| --- | ------ | --- | ------ | --- | --- | ------ | --- | ---------- | ----- | --- |
Figure 17.7: Deep ensemble with random priors. (a) Individual predictions from each member. Blue is the
fixed random prior function, orange is the trainable function, green is the combination of the two. (b) Overall
prediction from the ensemble, for increasingly large values of β. On the left we show (in red) the posterior
mean and pointwise standard deviation, and on the right we show samples from the posterior. As β increases,
we trust the random priors more, and pay less attention to the data, thus getting a more diffuse posterior.
| Generated | by randomized_priors.ipynb. |     |     |     |     |     |     |     |     |     |
| --------- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
where β 0 controls the amount of data-independent variation between the members. The trainable
network≥learns to model the residual error between the true output and the value predicted by the
prior. This is called a random prior deep ensemble [OAC18]. See Figure 17.7 for an illustration.
| 17.3.9.3 | Deep | ensembles |     | as approximate |     | Bayesian |     | inference |     |     |
| -------- | ---- | --------- | --- | -------------- | --- | -------- | --- | --------- | --- | --- |
The posterior predictive distribution for a Bayesian neural network cannot be expressed in closed
form. Therefore all Bayesian inference approaches in deep learning are approximate. In this context,
all approximate inference procedures fall onto a spectrum, representing how closely they approximate
the true posterior predictive distribution. Deep ensembles can provide better approximations to a
Bayesian model average than a single basin marginalization approach, because point masses from
differentbasinsofattractionrepresentgreaterfunctionaldiversitythanstandardBayesianapproaches
| which    | sample | within a  | single basin. |              |           |     |     |     |     |     |
| -------- | ------ | --------- | ------------- | ------------ | --------- | --- | --- | --- | --- | --- |
| 17.3.9.4 | Deep   | ensembles |               | vs classical | ensembles |     |     |     |     |     |
Note that deep ensembles are slightly different from classical ensemble methods (see e.g., [Die00]),
such as bagging and random forests, which obtain diversity of their predictors by training them on
different subsets of the data (created using bootstrap resampling), or on different features. This data
perturbation is necessary to get diversity when the base learner is a convex problem (such as a linear
model, or shallow decision tree). In the deep ensemble approach, every model is trained on the same
data, and the same input features. The diversity arises due to different starting parameters, different
| Author: | Kevin | P. Murphy. | (C) | MIT Press. | CC-BY-NC-ND |     |     | license |     |     |
| ------- | ----- | ---------- | --- | ---------- | ----------- | --- | --- | ------- | --- | --- |

660
|     |     |     |                  |     |                     |     |                      | Chapter | 17. Bayesian | neural | networks |     |
| --- | --- | --- | ---------------- | --- | ------------------- | --- | -------------------- | ------- | ------------ | ------ | -------- | --- |
|     |     |     | One shared       |     | ...multiplied by    |     | ...yields ensemble   |         |              |        |          |     |
|     |     |     | weight matrix    |     | independent rank    |     | weight matrices for  |         |              |        |          |     |
|     |     |     | (slow weight)... |     | one fast weights... |     | each member.         |         |              |        |          |     |
|     |     |     |                  |     |                     |     |                      | -1      | 0 1          |        |          |     |
W
Figure 17.8: Illustration of batch ensemble with 2 ensemble members. From Figure 2 of [WTB20]. Used with
| kind permission |     | of Paul | Vicol. |     |     |     |     |     |     |     |     |     |
| --------------- | --- | ------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
random seeds, and SGD noise, which induces different solutions due to the nonconvex loss. It is
also possible to explicitly enforce diversity of the ensemble members, which can provably improve
| performance | [TB22].  |              |     |             |             |            |     |              |                      |               |     |     |
| ----------- | -------- | ------------ | --- | ----------- | ----------- | ---------- | --- | ------------ | -------------------- | ------------- | --- | --- |
| 17.3.9.5    | Deep     | ensembles    |     | vs mixtures |             | of experts |     | and stacking |                      |               |     |     |
| If we use   | weighted | combinations |     | of          | the models, |            |     | M            |                      | ), where      |     | is  |
|             |          |              |     |             |             | p(θ        | )=  | p(m          | )p(θ m,              |               | p(m | )   |
|             |          |              |     |             |             | lar|gDe    |     | m= 1         | t|hDis mix|tureDwill | concentra|tDe |     |     |
the marginal likelihood of model m, then, in the sample lim it, on
the MAP model, so only one component will be selected. B(cid:80)y contrast, in deep ensembles, we always
use M equally weighted models. Thus we see that Bayes model averaging is not the same as model
ensembling [Min00b]. Indeed, ensembling can enlarge the expressive power of the posterior predictive
| distribution | compared |          | to BMA | [OCM21]. |     |                |     |                |     |     |         |     |
| ------------ | -------- | -------- | ------ | -------- | --- | -------------- | --- | -------------- | --- | --- | ------- | --- |
| We can       | also     | make the | mixing | weights  |     | be conditional |     | on the inputs: |     |     |         |     |
| p(y x,       | )=       | w        | (x)p(y | x,θ      | )   |                |     |                |     |     | (17.21) |     |
|              |          | m        |        | m        |     |                |     |                |     |     |         |     |
| |            | D        |          |        | |        |     |                |     |                |     |     |         |     |
m
(cid:88)
If we constrain the weights to be non-zero and sum to one, this is called a mixture of experts.
However,ifweallowageneralpositiveweightedcombination,theapproachiscalledstacking[Wol92;
Bre96; Yao+18a; CAII20]. In stacking, the weights w (x) are usually estimated on hold-out data,
m
| to make  | the method | more     | robust | to  | model | misspecification. |     |     |     |     |     |     |
| -------- | ---------- | -------- | ------ | --- | ----- | ----------------- | --- | --- | --- | --- | --- | --- |
| 17.3.9.6 | Batch      | ensemble |        |     |       |                   |     |     |     |     |     |     |
Deep ensembles require times more memory and time than a single model. One way to reduce
M
the memory cost is to share most of the parameters — which we call weights, — and then
|     |     |     |     |     |     |     |     |     | slow | W   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- |
let each ensemble member m estimate its own local perturbation, which we will call fast weights,
. We then define . For efficiency, we can define to be a rank-one matrix,
| F m |     |     | W m = | W   | F m |     |     |     | F m |     |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Fig⊙ure
F =s rT, as illustrated in 17.8. This is called batch ensemble [WTB20].
| m m | m   |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

661
| 17.3. | Posteriors |     | for BNNs |     |     |     |     |     |     |     |
| ----- | ---------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
It is clear that the memory overhead is very small compared to naive ensembles, since we just need
to store vectors (sl and ) for every layer l, which is negligible compared to the quadratic
|      |            | 2M  |            |        | rl     |     |     |     |     |     |
| ---- | ---------- | --- | ---------- | ------ | ------ | --- | --- | --- | --- | --- |
|      |            |     |            | m      | m      |     |     |     |     |     |
| cost | of storing |     | the shared | weight | matrix | Wl. |     |     |     |     |
In addition to memory savings, batch ensemble can reduce the inference time by a constant factor
by leveraging within-device parallelism. To see this, consider the output of one layer using ensemble
| m   | on example |     | n:   |     |       |     |       |     |       |         |
| --- | ---------- | --- | ---- | --- | ----- | --- | ----- | --- | ----- | ------- |
|     | ym         | WT  |      |     | rT)Tx |     | (WT(x |     |       | (17.22) |
|     | =φ         |     | x =φ | (W  | s     |     | =φ    |     | s ) r |         |
|     | n          |     | m n  |     | ⊙ m   | m n |       | n ⊙ | m ⊙ m |         |
We can vec(cid:0)torize th(cid:1)is for (cid:0)a minibatch of inp(cid:1)uts X(cid:0)by replicating r and(cid:1) s along the B rows in the
|       |     |      |           |        |     |     |     |     | m m |     |
| ----- | --- | ---- | --------- | ------ | --- | --- | --- | --- | --- | --- |
| batch | to  | form | matrices, | giving |     |     |     |     |     |     |
(17.23)
|     | Y =φ(((X |     | S )W) |     | R ) |     |     |     |     |     |
| --- | -------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
|     | m        |     | ⊙ m   | ⊙   | m   |     |     |     |     |     |
This applies the same ensemble parameters m to every example in the minibatch of size B. To
achieve diversity during training, we can divide the minibatch into sub-batches, and use sub-batch
M
m to train W . (Note that this reduces the batch size for training each ensemble to B/M.) At test
m
time, when we want to average over M models, we can replicate each input M times, leading to a
| batch | size | of BM. |     |     |     |     |     |     |     |     |
| ----- | ---- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
In [WTB20], they show that this method outperforms MC dropout at negligible extra memory
cost. However, the best combination was to combine batch ensemble with MC dropout; in some
| cases,  | this | approached    |     | the performance |     | of naive  | ensembles. |     |              |     |
| ------- | ---- | ------------- | --- | --------------- | --- | --------- | ---------- | --- | ------------ | --- |
| 17.3.10 |      | Approximating |     |                 | the | posterior | predictive |     | distribution |     |
Once we have approximated the parameter posterior, q(θ) p(θ ), we can use it to approximate
≈ |D
| the | posterior | predictive |     | distribution: |     |     |     |     |     |     |
| --- | --------- | ---------- | --- | ------------- | --- | --- | --- | --- | --- | --- |
(17.24)
|     | p(y x, | )=  | q(θ)p(y | x,θ)dθ |     |     |     |     |     |     |
| --- | ------ | --- | ------- | ------ | --- | --- | --- | --- | --- | --- |
|     | |      | D   |         | |      |     |     |     |     |     |     |
(cid:90)
| We  | often | approximate | this | integral |     | using Monte | Carlo: |     |     |     |
| --- | ----- | ----------- | ---- | -------- | --- | ----------- | ------ | --- | --- | --- |
S
1
|     | p(y x, | )   | p(y | f(x,θs)) |     |     |     |     |     | (17.25) |
| --- | ------ | --- | --- | -------- | --- | --- | --- | --- | --- | ------- |
|     | |      | D ≈ | S   | |        |     |     |     |     |     |         |
s=1
(cid:88)
| where | θs  | q(θ | ). We | discuss | some | extensions | of this | approach | below. |     |
| ----- | --- | --- | ----- | ------- | ---- | ---------- | ------- | -------- | ------ | --- |
∼ |D
| 17.3.10.1 |     | A   | linearized | approximation |     |     |     |     |     |     |
| --------- | --- | --- | ---------- | ------------- | --- | --- | --- | --- | --- | --- |
In [IKB21] they point out that samples from an approximate posterior, q(θ), can result in bad
predictions when plugged into the model if the posterior puts probability density “in the wrong
places”. Thisisbecausef(x;θ)isahighlynonlinearfunctionofθ thatmightbehavequitedifferently
when θ is far from the MAP estimate on which q(θ) is centered. To avoid this problem, they propose
to replace with a linear approximation centered at the MAP estimate :
f(x;θ) θ ∗
|     | f θ ∗(x,θ)=f(x,θ |     | )+J(x)(θ |     |     | θ ) |     |     |     | (17.26) |
| --- | ---------------- | --- | -------- | --- | --- | --- | --- | --- | --- | ------- |
|     | l in             |     | ∗        |     |     | ∗   |     |     |     |         |
−
| Author: |     | Kevin | P. Murphy. | (C) | MIT | Press. | CC-BY-NC-ND | license |     |     |
| ------- | --- | ----- | ---------- | --- | --- | ------ | ----------- | ------- | --- | --- |

662
|     |     |     |     |     |     | Chapter 17. | Bayesian neural | networks |
| --- | --- | --- | --- | --- | --- | ----------- | --------------- | -------- |
where ∂f(x;θ) is the Jacobian matrix, where is the number of parameters, and
|     | J   | θ (x)= |     | θ   | P C | P   |     |     |
| --- | --- | ------ | --- | --- | --- | --- | --- | --- |
is the n∗umber of ∂θ outp| u∗ts. Such a×model is well behaved around , and so the approximation
| C   |     |     |     |     |     | θ   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
∗
S
1
|     | p(y x, | )   | p(y | fθ∗(x,θs)) |     |     |     | (17.27) |
| --- | ------ | --- | --- | ---------- | --- | --- | --- | ------- |
|     |        | S   |     | lin        |     |     |     |         |
|     | |      | D ≈ |     | |          |     |     |     |         |
s=1
(cid:88)
| often | works | better | than | Equation | (17.25). |     |     |     |
| ----- | ----- | ------ | ---- | -------- | -------- | --- | --- | --- |
Note that fθ∗(x,θ) is a linear function of the parameters θ, but a nonlinear function of
z =
lin
the inputs x. Thus p(y fθ∗(x,θ)) is a generalized linear model (Section 15.1), so [IKB21] call this
li n
| approximation |     | the | GLM | | p redictive | distribution. |     |     |     |
| ------------- | --- | --- | --- | ------------- | ------------- | --- | --- | --- |
If we have a Gaussian approximation to the parameter, µ,Σ), then we can “push
|      |          |      |                    |               |        | p(θ ) | (θ  |         |
| ---- | -------- | ---- | ------------------ | ------------- | ------ | ----- | --- | ------- |
|      |          |      |                    |               |        | |D ≈N | |   |         |
| this | through” | the  | linear             | approximation | to get |       |     |         |
|      |          |      | f(x,µ),J(x)TΣJ(x)) |               |        |       |     | (17.28) |
|      | p(z x,   | )    | (z                 |               |        |       |     |         |
|      | |        | D ≈N | |                  |               |        |       |     |         |
where z are the logits. (Alternatively, we can use the last layer method of Equation (17.17) to get
a Gaussian approximation to p(z x, ).) If we approximate the final softmax layer with a probit
function, we can analytically pass|thisDGaussian through the final softmax layer to deterministically
compute the predictive probabilities p(y =cx, ), using Equation (15.150). Alternatively, we can
| use       | the | Laplace | bridge  | approximation | in Sect|ionD17.3.10.2. |     |     |     |
| --------- | --- | ------- | ------- | ------------- | ---------------------- | --- | --- | --- |
| 17.3.10.2 |     | The     | Laplace | bridge        | approximation          |     |     |     |
Justusingapointestimateoftheprobabilityofeachclasslabel,p =p(y =cx, ),canbeunreliable,
c
since it does not convey any sense of uncertainty in the probability value, ev|enDthough we may have
taken the uncertainty of the parameters into account (e.g., using the methods of Section 17.3.10.1).
An alternative is to represent the output over labels as a Dirichlet distribution, α), rather
Dir(π
appropriate|
than a categorical distribution, Cat(y p), where p=softmax(z). This is more if we view
wi|th
each datapoint as being annotated a “soft” vector of probabilities (e.g., representing consensus
votes from human raters), rather than a one-hot encoding with a single “ground truth” value. This
can be useful for settings where the true label is ambiguous (see e.g., [Bey+20; Dum+18]).
WecaneithertrainthemodeltopredicttheDirichletparametersdirectly(asintheprior network
approach of [MG18]), or we can train the model to predict softmax outputs in the usual way, and
then derive the Dirichlet parameters from a Gaussian approximation to the posterior. The latter
approach is known as the [HKH22], and has the advantage that it can be used as a
|     |     |     |     | Laplace | bridge |     |     |     |
| --- | --- | --- | --- | ------- | ------ | --- | --- | --- |
post-processing method. It works as follows. First we compute a Gaussian approximation to the
logits, using Equation (17.28) or Equation (17.17). Then we compute
|     | p(z | x, )= | (z  | m,V) |     |     |     |     |
| --- | --- | ----- | --- | ---- | --- | --- | --- | --- |
|     |     | | D   | N   | |    |     |     |     |     |
C
|     |     | 1        | 2   | exp(m ) |          |     |     |         |
| --- | --- | -------- | --- | ------- | -------- | --- | --- | ------- |
|     | α = | 1        | +   | i       | exp( m ) |     |     | (17.29) |
|     | i   |          |     |         | j        |     |     |         |
|     |     | V ii  − | C   | C2      | −       |     |     |         |
j=1
(cid:88)
 
where is the number of classes. We can then derive the probabilities of each class label using
C
| p   | =E[π | ]=α /α | , where | α = | C α . |     |     |     |
| --- | ---- | ------ | ------- | --- | ----- | --- | --- | --- |
| c   |      | c c    | 0       | 0   | c=1 c |     |     |     |
Note that the derivation of the above result assumes that the Gaussian terms sum to zero, since
o(cid:80)f
the Gaussian has one less degree freedom compared to the Dirichlet. To ensure this, it is necessary
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

663
| 17.3. | Posteriors |     | for BNNs    |     |     |                 |     |     |     |                     |                   |     |
| ----- | ---------- | --- | ----------- | --- | --- | --------------- | --- | --- | --- | ------------------- | ----------------- | --- |
|       |            |     | mouse       |     |     | desktopcomputer |     |     |     | hometheater         | modem             |     |
|       |            |     | loupe       |     |     |                 |     |     |     | entertainmentcenter |                   |     |
|       |            |     |             |     |     | spacebar        |     |     |     |                     | hand-heldcomputer |     |
|       |            |     | desk        |     |     |                 |     |     |     | desktopcomputer     |                   |     |
|       |            |     | hairspray   |     |     | laptop          |     |     |     | television          | spacebar          |     |
|       |            |     | groom       |     |     | modem           |     |     |     | notebook            | laptop            |     |
|       |            |     | facepowder  |     |     | notebook        |     |     |     | monitor             | notebook          |     |
|       |            |     | stethoscope |     |     |                 |     |     |     | screen              |                   |     |
labcoat
notebook
0.00 0.01 0.02 0.03 0.04 0.05 0.00 0.05 0.10 0.15 0.20 0.25 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.0 0.2 0.4 0.6 0.8 1.0
Figure 17.9: Illustration of uncertainty about individual labels in an image classification problem. Top row:
images from the “laptop” class of ImageNet. Bottom row: beta marginals for the top-k predtions for the
respective image. First column: high uncertainty about all the labels. Second column: “notebook” and “laptop”
have high confidence. Third column: “desktop”, “screen” and “monitor” have high confidence. Fourth column:
only “laptop” has high confidence. (Compare to Figure 14.4.) From Figure 6 of [HKH22]. Used with kind
| permission |     | of Philipp | Hennig. |     |     |     |     |     |     |     |     |     |
| ---------- | --- | ---------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
to first project the Gaussian distribution onto this constraint surface, yielding
|     |        |     |     | V11Tm |     |     | V11TV |     |     |      |     |         |
| --- | ------ | --- | --- | ----- | --- | --- | ----- | --- | --- | ---- | --- | ------- |
|     | p(z x, | )=  | z   | m     |     | ,V  |       | =   | (z  | m,V) |     | (17.30) |
|     | |      | D   | N | | −     | 1TV | −   | 1TV1  |     | N | | ′ ′  |     |         |
1
|     |     |     | (cid:18) |     | ∗   |     |     | (cid:19) |     |     |     |     |
| --- | --- | --- | -------- | --- | --- | --- | --- | -------- | --- | --- | --- | --- |
where1istheonesvectorofsizeC. Toavoidpotentialproblemswhereαissparse, [HKH22]propose
to also scale the posterior (after the zero-sum constraint) by using m =m/√c and V =V/c,
|       |     |     |      |      |     |     |     |     |     | ′′ ′ |     | ′′ ′ |
| ----- | --- | --- | ---- | ---- | --- | --- | --- | --- | --- | ---- | --- | ---- |
| where | c=( |     | V )/ | C/2. |     |     |     |     |     |      |     |      |
i′i
One useful property ii of the Laplace bridge approximation, compared to the probit approximation,
isthatweca(cid:80)neasilyc(cid:112)omputeamarginaldistributionovertheprobablilityofeachlabelbeingpresent.
This is because the marginals of a Dirichlet are beta distributions. We can use this to adaptively
compute a top-k prediction set; this is similar in spirit to conformal prediction (Section 14.3.1), but
is Bayesian, in the sense that it represents per-instance uncertainty. The method works as follows.
First we sort the class labels in decreasing order of expected probability, to get α˜; next we compute
| the | marginal | distribution |     | over | the | probability | for | the | top label, |     |     |     |
| --- | -------- | ------------ | --- | ---- | --- | ----------- | --- | --- | ---------- | --- | --- | --- |
(17.31)
|     | p(π 1 x, | )=Beta(α˜ |     | 1 ,α 0 | α˜ 1 ) |     |     |     |     |     |     |     |
| --- | -------- | --------- | --- | ------ | ------ | --- | --- | --- | --- | --- | --- | --- |
|     | |        | D         |     |        | −      |     |     |     |     |     |     |     |
where α = α . We then compute the marginal distributions for the other labels in a similar way,
|     | 0   | c   | c   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:80)
| Author: | Kevin |     | P. Murphy. |     | (C) MIT | Press. | CC-BY-NC-ND |     |     | license |     |     |
| ------- | ----- | --- | ---------- | --- | ------- | ------ | ----------- | --- | --- | ------- | --- | --- |

664
|     |     |     |     |     |     |     |     | Chapter | 17. Bayesian neural | networks |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------------- | -------- |
and return all labels that have significant overlap with the top label. As we see from the examples in
Figure 17.9, this approach can return variable-sized outputs, reflecting uncertainty in a natural way.
| 17.3.10.3 |     | Distillation |     |     |     |     |     |     |     |     |
| --------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
The MC approximation to the posterior predictive is S times slower than a standard, deterministic
plug-in approximation. One way to speed this up is to use distillation to approximate the
semi-parametric “teacher” model from Equation (17.25) by a parametric “student” model
|     |     |     |     |     | p   |     |     |     |     | p   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     | t   |     |     |     | s   |
by minimizing E[DKL(p (y x) p (y x))] wrt p . This approach was first proposed in [HVD14],
|     |     |     |     | t   | s   |     | s   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
who called the technique “d | ∥ | ledge”, because the teacher has “hidden” information in its
|     |     |     |     | ark | know |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
predictive probabilities (logits) than is not apparent in the raw one-hot labels.
In [Kor+15], this idea was used to distill the predictions from a teacher whose parameter posterior
was computed using HMC; this is called “Bayesian knowledge”. A similar idea was used in
dark
[BPK16; GBP18], who distilled the predictive distribution derived from MC dropout (Section 17.3.1).
Since the parametric student is typically less flexible than the semi-parametric teacher, it may be
overconfident, and lack diversity in its predictions. To avoid this overconfidence, it is safer to make
| the     | student | be       | a mixture | distribution |      | [SG05;     | Tra+20a]. |     |     |     |
| ------- | ------- | -------- | --------- | ------------ | ---- | ---------- | --------- | --- | --- | --- |
| 17.3.11 |         | Tempered |           | and          | cold | posteriors |           |     |     |     |
When working with BNNs for classification problems, the likelihood is usually taken to be
|     | p(y x,θ)=Cat(y |     | softmax(f(x;θ))) |     |     |     |     |     |     | (17.32) |
| --- | -------------- | --- | ---------------- | --- | --- | --- | --- | --- | --- | ------- |
|     | |              |     | |                |     |     |     |     |     |     |         |
where returns the logits over the class labels. This is the same as in multinomial
|     | f(x;θ) |     | R C |     |     |     | C   |     |     |     |
| --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
regress∈ion
logistic (Section 15.3.2); the only difference is that f is a nonlinear function of θ.
However, in practice, it is often found (see e.g., [Zha+18; Wen+20b; LST21; Noc+21]) that BNNs
give better predictive accuracy if the likelihood function is scaled by some power α. That is, instead
of targeting the posterior p(θ ) p(y x,θ)p(θ), these methods target the tempered posterior,
|          |      |      |           |         | | D ).∝In          | log|space, | we  | have |     |         |
| -------- | ---- | ---- | --------- | ------- | ------------------ | ---------- | --- | ---- | --- | ------- |
| p        | (θ   | )    | p(y       | X,θ)αp( | θ                  |            |     |      |     |         |
| tempered |      | |D ∝ | |         |         |                    |            |     |      |     |         |
|          | logp | (θ   | )=αlogp(y |         | X,θ)+logp(θ)+const |            |     |      |     | (17.33) |
tempered
|      |         |        | |D  |             | |   |       |           |           |     |     |
| ---- | ------- | ------ | --- | ----------- | --- | ----- | --------- | --------- | --- | --- |
| This | is also | called | an  |             |     | or    |           | [Med+21]. |     |     |
|      |         |        |     | α-posterior |     | power | posterior |           |     |     |
Another common method is to target the cold posterior, p (θ ) p(θ X,y)1/T, or, in log
cold
| space, |      |     |     |        |       |               |     |     | |D ∝ | |         |
| ------ | ---- | --- | --- | ------ | ----- | ------------- | --- | --- | ------ | ------- |
|        |      |     | 1   |        |       | 1             |     |     |        |         |
|        | logp | (θ  | )=  | logp(y | X,θ)+ | logp(θ)+const |     |     |        | (17.34) |
|        | cold | |D  | T   |        | |     | T             |     |     |        |         |
If T <1, we say that the posterior is “cold”. Note that, in the case of a Gaussian prior, using the
cold posterior is the same as using the tempered posterior with a different hyperparameter, since
|     |      | is  | given | by  |     |     |     |     |     |     |
| --- | ---- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
| 1   | logp | (θ) |       |     |     |     |     |     |     |     |
T cold
|     | 1   |      |          |       | 1    |           |     |             |          |         |
| --- | --- | ---- | -------- | ----- | ---- | --------- | --- | ----------- | -------- | ------- |
|     |     | 0,σ2 |          |       |      | θ2+const= |     | 0,σ2        |          | (17.35) |
|     | log | (θ   | cold I)= |       |      |           |     | (θ tempered | I)+const |         |
|     | T   | N |  |          | −2Tσ2 |      | i         |     | N |         |          |         |
|     |     |      |          |       | cold | i         |     |             |          |         |
(cid:88)
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

17.4. Generalization in Bayesian deep learning 665
Figure 17.10: Flat vs sharp minima. From Figures 1 and 2 of [HS97]. Used with kind permission of Jürgen
Schmidhuber.
which equals logp (θ) if we set σ2 = Tσ2 . Thus both methods are effectively the
tempered tempered cold
same, and just reweight the likelihood by α=1/T.
Cold posteriors in Bayesian neural network classifiers are a consequence of underrepresenting
aleatoric (label) uncertainty, as shown by [Kap+22]. On benchmarks such as CIFAR-100, we should
have essentially no uncertainty about the labels of the training images, yet Bayesian classifiers with
softmaxlikelihoodshaveveryhighuncertaintyforthesepoints. Moreover,[Izm+21b]showedthatthe
cold posterior effect in all the examples of [Wen+20b] when data augmentation is removed. [Kap+22]
show that with the SGLD inference in [Wen+20b], data augmentation has the effect of raising the
likelihood to a power 1/K for minibatches of size K. Cold posteriors exactly counteract this effect,
more honestly representing our beliefs about aleatoric uncertainty, by sharpening the likelihood.
However, tempering is not required, and [Kap+22] show that by using a Dirichlet observation
model to explicitly represent (lack of) label noise, there is no cold posterior effect, even with data
augmentation. The curation hypotheses of [Ait21] can be considered a special case of the above
explanation, where curation has the effect of increasing our confidence about training labels.
In Section 14.1.3, we discuss generalized variational inference, which gives a general framework for
understanding whether and how the likelihood or prior could benefit from tempering. Tempering is
particularly useful if (as is usually the case) the model is misspecified [KJD21].
17.4 Generalization in Bayesian deep learning
In this section, we discuss why “being Bayesian” can improve predictive accuracy and generalization
performance.
17.4.1 Sharp vs flat minima
Some optimization methods (in particular, second-order batch methods) are able to find “needles
in haystacks”, corresponding to narrow but deep “holes” in the loss landscape, corresponding to
parameter settings with very low loss. These are known as sharp minima, see Figure 17.10(right).
From the point of view of minimizing the empirical loss, the optimizer has done a good job. However,
such solutions generally correspond to a model that has overfit the data. It is better to find points
that correspond to flat minima, as shown in Figure 17.10(left); such solutions are more robust and
Author: Kevin P. Murphy. (C) MIT Press. CC-BY-NC-ND license

666 Chapter 17. Bayesian neural networks
generalize better. To see why, note that flat minima correspond to regions in parameter space where
there is a lot of posterior uncertainty, and hence samples from this region are less able to precisely
memorize irrelevant details about the training set [AS17]. Put another way, the description length for
sharp minima is large, meaning you need to use many bits of precision to specify the exact location
in parameter space to avoid incurring large loss, whereas the description length for flat minima is
less, resulting in better generalization [Mac03].
SGD often finds such flat minima by virtue of the addition of noise, which prevents it from
“entering” narrowregionsofthelosslandscape(seeSection12.5.7). Inaddition,inhigherdimensional
spaces, flat regions occupy a much greater volume, and are thus much more easily discoverable by
optimization procedures. More precisely, the analysis in [SL18] shows that the probability of entering
any given basin of attraction around a minimum is given by p (θ ) e (θ)dθ. Note
SGD −L
that this is integrating over thAe volume of space corresponding to , an∈d Ahen∝ce iAs proportional to
the model evidence (marginal likelihood) for that region, as explaAined in Sectio(cid:82)n 3.8.1. Since the
evidence is parameterization invariant (since we marginalize out the parameters), this means that
SGD will avoid regions that have low evidence (corresponding to sharp minima) regardless of how we
parameterize the model (contrary to the claims in [Din+17]).
In fact, several papers have shown that we can view SGD as approximately sampling from the
Bayesian posterior (see Section 17.3.8). The SWA method (Section 17.3.8) can be seen as finding a
center of mass in the posterior based on these SGD samples, finding solutions that generalize better
than picking a single SGD point.
If we must use a single solution, a flat one will help us better approximate the Bayesian model
average in the integral of Equation (17.1). However, by attempting to perform a more complete
Bayesian model average, we will select for flatness without having to deal with the messiness of
having to worry about flatness definitions, or the effects of reparameterization, or unknown implicit
regularization, as the model average will automatically weight regions with the greatest volume.
17.4.2 Mode connectivity and the loss landscape
In DNNs there are often many low-loss solutions, which provide complementary explanations of
the data. Moreover, in [Gar+18c] they showed that two independently trained SGD solutions can
be connected by a curve in a subspace, along which the training loss remains near-zero, known as
mode connectivity. Despite having the same training loss, these different parameter settings give
rise to very different functions, as illustrated in Figure 17.11, where we show predictions on a 1d
regression problem coming from different points in parameter space obtained by interpolating along
a mode connecting curve between two distinct MAP estimates. Using a Bayesian model average, we
can combine these functions together to provide much better performance over a single flat solution
[Izm+19].
Recently, it has been discovered [Ben+21b] that there are in fact large multidimensional simplexes
of low loss solutions, which can be combined together for significantly improved performance. These
results further motivate the Bayesian approach (Equation (17.1)), where we perform a posterior
weighted model average.
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

17.4. Generalization in Bayesian deep learning 667
Figure17.11: Diversityofhighperformingfunctionssampledfromtheposterior. Toprow: weshowpredictions
on the 1d input domain for 4 different functions. We see that they extrapolate in different ways outside of the
support of the data. Bottom row: we show a 2d subspace spanning two distinct modes (MAP estimates), and
connected by a low-loss curved path computed as in [Gar+18c]. From Figure 8 of [WI20]. Used with kind
permission of Andrew Wilson.
17.4.3 Effective dimensionality of a model
Modern DNNs have millions of parameters, but these parameters are often not well-determined
by the data, i.e., there can be a lot of posterior uncertainty. By averaging over the posterior, we
reduce the chance of overfitting, because we do not use “degrees of freedom” that are not needed or
warranted.
To quantify the number of degrees of freedom, or effective dimensionality [Mac92b], we follow
[MBW20] and define
k
λ
N (H,c)= i , (17.36)
eff
λ +c
i
i=1
(cid:88)
where λ are the eigenvalues of the Hessian matrix H computed at a local mode, and c > 0 is a
i
regularization parameter. Intuitively, the effective dimension counts the number of well-determined
parameters. A “flat minimum” will have many directions in parameter space that are not well-
determined, and hence will have low effective dimensionality. This means that we can perform
Bayesian inference in a low dimensional subspace [Izm+19]: Since there is functional homogeneity
in all directions but those defining the effective dimension, neural networks can be significantly
compressed.
This compression perspective can also be used to understand why the effective dimension can be a
good proxy for generalization. If two models have similar training loss, but one has lower effective
dimension, then it is providing a better compression for the data at the same fidelity. In Figure 17.12
weshowthatforCNNswithlowtrainingloss(abovethegreenpartition), theeffectivedimensionality
closely tracks generalization performance. We also see that the number of parameters alone is not a
strong determinant of generalization. Indeed, models with more parameters can have a lower number
of effective parameters. We also see that wide but shallow models overfit, while depth helps provide
Author: Kevin P. Murphy. (C) MIT Press. CC-BY-NC-ND license

668
|     |                          |     |       |           | Chapter | 17. Bayesian | neural     | networks |
| --- | ------------------------ | --- | ----- | --------- | ------- | ------------ | ---------- | -------- |
| 8   | Effective Dimensionality |     | 100 8 | Test Loss |         | 2.2 8        | Train Loss |          |
3.5
|       |     |     | 95   |     |     | 2.0   |     | 3.0 |
| ----- | --- | --- | ---- | --- | --- | ----- | --- | --- |
| 6     |     |     | 6    |     |     | 6     |     |     |
|       |     |     | 90   |     |     |       |     | 2.5 |
| htpeD |     |     |      |     |     | 1.8   |     |     |
| 4     |     |     | 85 4 |     |     | 4     |     | 2.0 |
|       |     |     | 80   |     |     | 1.6   |     | 1.5 |
| 2     |     |     | 75 2 |     |     | 1.4 2 |     | 1.0 |
|       |     |     | 70   |     |     |       |     | 0.5 |
| 0     |     |     | 0    |     |     | 1.2 0 |     |     |
12 16 20 24 28 32 36 12 16 20 24 28 32 36 12 16 20 24 28 32 36
|     | Width |     |     | Width |     |     | Width |     |
| --- | ----- | --- | --- | ----- | --- | --- | ----- | --- |
Figure 17.12: Left: effective dimensionality as a function of model width and depth for a CNN on CIFAR-100.
Center: test loss as a function of model width and depth. Right: train loss as a function of model width and
depth. Yellow level curves represent equal parameter counts (1e5, 2e5, 4e5, 1.6e6). The green curve separates
models with near-zero training loss. Effective dimensionality serves as a good proxy for generalization for
models with low train loss. We see wide but shallow models overfit, providing low train loss, but high test
loss and high effective dimensionality. For models with the same train loss, lower effective dimensionality
can be viewed as a better compression of the data at the same fidelity. Thus depth provides a mechanism for
compression, which leads to better generalization. From Figure 2 of [MBW20]. Used with kind permission of
Andrew Wilson.
lower effective dimensionality, leading to a better compression of the data. It is depth that makes
modern neural networks distinctive, providing hierarchical inductive biases making it possible to
| discover | more regularity | in the | data.   |     |     |     |     |     |
| -------- | --------------- | ------ | ------- | --- | --- | --- | --- | --- |
| 17.4.4   | The hypothesis  | space  | of DNNs |     |     |     |     |     |
Zhang et al. [Zha+17] showed that CNNs can fit CIFAR-10 images with random labels with zero
training error, but can still generalize well on the noise-free test set. It has been claimed that this
result contradicts a classical understanding of generalization, because it shows that neural networks
are capable of significantly overfitting the data, but can still generalize well on structured inputs.
We can resolve this paradox by taking a Bayesian perspective. In particular, we know that
modern CNNs are very flexible, so they can fit almost any pattern (since they are in fact universal
approximators). However, their architecture encodes a prior over what kinds of patterns they expect
to see in the data (see Section 17.2.5). Image datasets with random labels can be represented by this
function class, but such solutions receive very low marginal likelihood, since they strongly violate the
prior assumptions [WI20]. By contrast, image datasets where the output labels are consistent with
| patterns | in the input | get much | higher marginal | likelihood. |     |     |     |     |
| -------- | ------------ | -------- | --------------- | ----------- | --- | --- | --- | --- |
This phenomenon is not unique to DNNs. For example, it also occurs with Gaussian processes
(Chapter18). Suchmodelsarealsouniversalapproximators,buttheyallocatemostoftheirprobability
mass to a small range of solutions (depending on the chosen kernel). They can also fit image datasets
with random labels, but such data receives a low marginal likelihood [WI20].
In general, we can distinguish the support of a model, i.e., the set of functions it can represent,
fromthedistributionoverthatsupport, i.e., theinductivebiaswhichleadsittoprefersomefunctions
overothers. Wewouldliketousemodelswherethesupportislarge,sowecancapturethecomplexity
of real-world data, but also where the inductive bias places probability mass on the kinds of functions
we expect to see. If we succeed at this, the posterior will quickly converge on the true function after
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

669
| 17.4. Generalization |     | in  | Bayesian | deep learning |     |
| -------------------- | --- | --- | -------- | ------------- | --- |
Figure 17.13: Illustration of the behavior of different kinds of model families and the prior distributions
they induce over datasets. (a) The purple model is a simple linear model that has small support, and can
only represent a few kinds of datasets. The pink model is an unstructured MLP: this has support over a
large range of datasets with a fairly uninformative (broad) prior. Finally the green model is a CNN; this has
support over a large range of datasets but the prior is more concentrated on certain kinds of datasets that have
compositional structure. (b) The posterior for the green model (CNN) rapidly collapses to the true model,
since it is consistent with the data. (c) The posterior for the purple model (linear) also rapidly collapses, but
to a solution which cannot represent the true model. (d) The posterior for the pink model (MLP) collapses
very slowly (as a function of dataset size). From Figure 2 of [WI20]. Used with kind permission of Andrew
Wilson.
| seeing a small | amount    | of  | data. This | idea is sketched | in Figure 17.13. |
| -------------- | --------- | --- | ---------- | ---------------- | ---------------- |
| 17.4.5         | PAC-Bayes |     |            |                  |                  |
PAC-Bayes [McA99; LC02; Gue19; Alq21; GSZ21] provides a promising mechanism to derive
non-vacuous generalization bounds for large [Ney+17; NBS18; DR17], with
|     |     |     |     | stochastic | networks |
| --- | --- | --- | --- | ---------- | -------- |
parameters sampled from a probability distribution. In particular, the difference between the train
| error and | the generalization |     | error | can be expressed | as  |
| --------- | ------------------ | --- | ----- | ---------------- | --- |
| DKL(Q     | P)+c               |     |       |                  |     |
∥ , (17.37)
| (cid:115) 2(N | 1)  |     |     |     |     |
| ------------- | --- | --- | --- | --- | --- |
−
where c is a constant, N is the number of training points, P is the prior distribution over the
parameters, and is an arbitrary distribution, which can be chosen to optimize the bound.
Q
The perspective in this chapter is largely complementary, and in some ways orthogonal, to the
PAC-Bayes literature. Our focus has been on Bayesian marginalization, particularly multi-modal
marginalization, and a prescriptive approach to model construction. In contrast, PAC-Bayes bounds
are about bounding the empirical risk of a single sample, rather than marginalization, and are not
currently prescriptive: what we would do to improve the bounds, such as reducing the number
of model parameters, or using highly compact priors, does not typically improve generalization.
Moreover, while we have seen Bayesian model averaging over multimodal posteriors has a significant
effect on generalization, it has a minimal logarithmic effect on PAC-Bayes bounds. In general,
becausetheboundsareloose, albeitnon-vacuousinsomecases, thereisoftenroomtomakemodeling
choices that improve PAC-Bayes bounds without improving generalization, making it hard to derive
| a prescription | for | model   | construction | from the bounds.   |         |
| -------------- | --- | ------- | ------------ | ------------------ | ------- |
| Author: Kevin  | P.  | Murphy. | (C) MIT      | Press. CC-BY-NC-ND | license |

670
|     |     |     |     |     |     | Chapter 17. | Bayesian neural | networks |
| --- | --- | --- | --- | --- | --- | ----------- | --------------- | -------- |
0.9
5%
ycaruccA 0.8
0.7
| 0.6 |         |     | 25% |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | MAP BNN |     |     |     |     |     |     |     |
0.4
| Noshift | 1 2                   | 3 4 5 |     |         |         |      |                 |      |
| ------- | --------------------- | ----- | --- | ------- | ------- | ---- | --------------- | ---- |
|         | CorruptionIntensity   |       |     | 0.25    | 0.00    | 0.25 | 0.05 0.00       | 0.05 |
|         |                       |       |     | −       |         |      | −               |      |
| (a)     | ResNet-20, CIFAR-10-C |       |     | (b) BNN | weights |      | (c) MAP weights |      |
Figure 17.14: Bayesian neural networks under covariate shift. a: Performance of a ResNet-20 on the pixelate
corruption in CIFAR-10-C. For the highest degree of corruption, a Bayesian model average underperforms a
MAPsolutionby25%(44%against69%)accuracy. SeeIzmailovetal.[Izm+21b]fordetails. b: Visualization
of the weights in the first layer of a Bayesian fully-connected network on MNIST sampled via HMC. c: The
corresponding MAP weights. We visualize the weights connecting the input pixels to a neuron in the hidden
layer as a 28 28 image, where each weight is shown in the location of the input pixel it interacts with. This
×
| is Figure | 1 of Izmailov       | et al. [Izm+21a]. |                |     |     |      |     |     |
| --------- | ------------------- | ----------------- | -------------- | --- | --- | ---- | --- | --- |
| 17.4.6    | Out-of-distribution |                   | generalization |     | for | BNNs |     |     |
Bayesian methods are often assumed to be more robust in the context of distribution shift (discussed
in Chapter 19), because they capture more uncertainty than methods based on point estimation.
| However, | there are some | subtleties, | some    | of which | we discuss | below. |     |     |
| -------- | -------------- | ----------- | ------- | -------- | ---------- | ------ | --- | --- |
| 17.4.6.1 | BMA can        | give poor   | results | with     | default    | priors |     |     |
Many approximate inference methods, especially deep ensembles, are significantly less overconfident
(more well calibrated) in the presence of some kinds of covariate shifts [Ova+19]. However, in
[Izm+21b], it was noted that HMC, which arguably offers the most accurate approximation to the
| posterior, | often works | poorly under | distribution |     | shift. |     |     |     |
| ---------- | ----------- | ------------ | ------------ | --- | ------ | --- | --- | --- |
Rather than an idiosyncracy of HMC, Izmailov et al. [Izm+21a] show this lack of robustness
is a foundational issue of Bayesian model averaging under covariate shift, caused by degeneracies
in the training data, and a poor choice of prior. As an illustrative special case, MNIST digits all
have black corner pixels. Weights in the first layer of a neural network connected to these pixels
are multiplied by zero, and thus can take any value without affecting the outputs of the network.
Classical MAP training or deep ensembles of MAP solutions with a Gaussian prior will therefore
drive these parameters to zero, since they don’t help with the data fit, and the resulting network
will be robust to corruptions on these pixels. On the other hand, the posterior for these parameters
will be the same as the prior, and so a Bayesian model average will multiply corruptions by random
numbers sampled from the prior, leading to degraded predictive performance.
Figure 17.14(b, c) visualizes this example, showing the first-layer weights of a fully-connected
network for the MAP solution and a BNN posterior sample, on MNIST. The MAP weights corre-
sponding to zero intensity pixels near the boundary are near zero, while the BNN weights look noisy,
| sampled | from a Gaussian | prior. |     |     |     |     |     |     |
| ------- | --------------- | ------ | --- | --- | --- | --- | --- | --- |
Izmailov et al. [Izm+21a] prove that this issue is a special case of a much more general problem,
whenever there are linear dependencies in the input features of the training data, both for fully-
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

671
| 17.4. | Generalization |     | in  | Bayesian | deep | learning |     |     |     |     |
| ----- | -------------- | --- | --- | -------- | ---- | -------- | --- | --- | --- | --- |
connected and convolutional networks. In this case, the data live on a hyperplane. If a covariate or
domain shift, moves orthogonal to this hyperplane, the posterior will be the same as the prior in
the direction of the shift. The posterior model average will thus be highly vulnerable to shifts that
do not particularly affect the underlying semantic structure of the problem (such as corruptions),
| whereas |     | the MAP | solution | will | be entirely | robust | to such | shifts. |     |     |
| ------- | --- | ------- | -------- | ---- | ----------- | ------ | ------- | ------- | --- | --- |
By introducing a prior over parameters which is aligned with the principal components of the
traininginputs, wecansubstantiallyimprovethegeneralizationaccuracyofBayesianneuralnetworks
in out-of-distribution settings. Izmailov et al. [Izm+21a] propose the following EmpCov prior:
p(w1) (0,αΣ+ϵI), where w1 are the first layer weights, 1 n xT is the empirical
|             | =   |     |     |     |     |     |     | Σ   | =   | x i     |
| ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- |
| covariancNe |     |     |     |     |     |     |     |     | n 1 | i = 1 i |
of the training input features x , α>0 determines the sca l−e of t h e pr io r, and ϵ is a small
i
positive constant to ensure the covariance matrix is positive definite. Wi(cid:80)th this improved prior they
are able to obtain a method that is much more robust to distribution shift.
| 17.4.6.2 |     | BNNs | can | be overconfident |     | on  | OOD | inputs |     |     |
| -------- | --- | ---- | --- | ---------------- | --- | --- | --- | ------ | --- | --- |
Animportantprobleminpracticeishowapredictivemodelwillbehavewhenitisgivenaninputthat
is “out of distribution” or OOD. Ideally we would like the model to express that it is not confident
in its prediction, so that the system can abstain from predicting (see Section 19.3.3). Using “exact”
inference methods, such as MCMC, for BNNs can give this behavior in some cases. For example,
in Section 19.3.3.1 we showed that an MLP which was fit to MNIST using SGLD would be less
overconfident than a point estimate (computed using SGD) when presented with inputs from fashion
| MNIST. |     | However, | this | behavior | does not | always | occur | reliably. |     |     |
| ------ | --- | -------- | ---- | -------- | -------- | ------ | ----- | --------- | --- | --- |
To illustrate the problem, consider the 2d nonlinear binary classification dataset shown in Fig-
ure 17.15. In addition to the two training classes, we have highlighted (in green) a set of OOD inputs
that are far from the support of the training set. Intuitively we would expect the model to predict
a probability of 0.5 (corresponding to “don’t know”) for such inputs that are far from the training
set. However we see that the only methods that do so are the Gaussian process (GP) classifier (see
Section 18.4) and the SNGP model (Section 17.3.6), which contains a GP layer on top of the feature
extractor.
The lesson we learn from this simple example is that “being Bayesian” only helps if we are using a
good hypothesis class. If we only consider a single MLP classifier, with standard Gaussian priors on
the weights, it is extremely unlikely that we will learn the kind of compact decision boundary shown
in Figure 17.15g, because that function has negligible support under our prior (c.f. Section 17.4.4).
Instead we should embrace the power of Bayes to avoid overfitting and use as complex a model class
| as     | we can | afford. |           |     |          |     |     |     |     |     |
| ------ | ------ | ------- | --------- | --- | -------- | --- | --- | --- | --- | --- |
| 17.4.7 |        | Model   | selection |     | for BNNs |     |     |     |     |     |
Historically, the marginal likelihood (aka Bayesian evidence) has been used for model selection
problems, such as choosing neural architectures or hyperparameter values [Mac92a]. Recent methods
based on the Laplace approximation, such as [Imm+21; Dax+21], have made this scalable to large
BNNs. However, [Lot+22] argue that it is much better to use the conditional marginal likelihood,
| which   | we  | discuss | in Section | 3.8.5. |            |             |     |         |     |     |
| ------- | --- | ------- | ---------- | ------ | ---------- | ----------- | --- | ------- | --- | --- |
| Author: |     | Kevin   | P. Murphy. | (C)    | MIT Press. | CC-BY-NC-ND |     | license |     |     |

672
|     |     |     |     |     |     |     | Chapter | 17. | Bayesian | neural | networks |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | -------- | ------ | -------- |
1.05
| 2   |     |     |     | 2   |     | 2   |     |     | 2   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.90
| 1   |     |     | 0.75 | 1   |     | 1   |     |     | 1   |     |     |
| --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
0.60
| 0   |     |     |       | 0   |     | 0   |     |     | 0   |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | 0.45  | 1   |     | 1   |     |     | 1   |     |     |
| − 1 |     | 0   | 0.30− |     |     | −   |     |     | −   |     |     |
| 2   |     | 1   |       | 2   |     | 2   |     |     | 2   |     |     |
| −   |     |     | 0.15− |     |     | −   |     |     | −   |     |     |
OOD
| 3   |          |     | 0.00 | 3   |        | 3     |        |         | 3   |               |     |
| --- | -------- | --- | ---- | --- | ------ | ----- | ------ | ------- | --- | ------------- | --- |
| −   | 2 0      | 2   | 4    | − 2 | 0 2    | 4 − 2 | 0      | 2       | 4 − | 2 0           | 2 4 |
| −   |          |     |      | −   |        | −     |        |         |     | −             |     |
|     | (a)      | SGD |      | (b) | DE     |       | (c) MC | Dropout |     | (d) Bootstrap |     |
| 2   |          |     |      | 2   |        | 2     |        |         | 2   |               |     |
| 1   |          |     |      | 1   |        | 1     |        |         | 1   |               |     |
| 0   |          |     |      | 0   |        | 0     |        |         | 0   |               |     |
| 1   |          |     |      | 1   |        | 1     |        |         | 1   |               |     |
| −   |          |     |      | −   |        | −     |        |         | −   |               |     |
| 2   |          |     |      | 2   |        | 2     |        |         | 2   |               |     |
| −   |          |     |      | −   |        | −     |        |         | −   |               |     |
| − 3 |          |     |      | − 3 |        | − 3   |        |         | − 3 |               |     |
|     | 2 0      | 2   | 4    | 2   | 0 2    | 4 2   | 0      | 2       | 4   | 2 0           | 2 4 |
|     | −        |     |      | −   |        | −     |        |         |     | −             |     |
|     | (e) MCMC |     |      |     | (f) VI |       | (g)    | GP      |     | (h) SNGP      |     |
Figure 17.15: Predictions made by various (B)NNs when presented with the training data shown in blue and
red. The green blob is an example of some OOD inputs. Methods are: (a) standard SGD; (b) deep Ensemble
of 10 models with different random initializations; (c) MC dropout with 50 samples; (d) bootstrap training,
where each of the 10 models is initialized identically but given different versions of the data, obtained by
resampling with replacement; (e) MCMC using NUTS algorithm with 3000 warmup steps and 3000 samples;
(f) variational inference; (g) Gaussian process classifier using RBF kernel; (h) SNGP. The model is an MLP
with 8,16,16,8 units in the hidden layers and ReLu activation. The output layer has 1 neuron with sigmoid
| activation. | Generated |     | by makemoons_comparison.ipynb |     |     |     |     |     |     |     |     |
| ----------- | --------- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17.5        | Online    |     | inference                     |     |     |     |     |     |     |     |     |
In Section 17.3, we have focused on batch or offline inference. However, an important application of
Bayesian inference is in sequential settings, where the data arrives in a continuous stream, and the
model has to “keep up”. This is called inference, and is one approach to
|     |     |     |     |     | sequential | Bayesian |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------- | -------- | --- | --- | --- | --- | --- |
online learning (see Section 19.7.5). In this section, we discuss some algorithmic approaches to this
problem in the context of DNNs. These methods are widely used for continual learning, which we
| discuss | Section    | 19.7. |         |     |          |     |     |     |     |     |     |
| ------- | ---------- | ----- | ------- | --- | -------- | --- | --- | --- | --- | --- | --- |
| 17.5.1  | Sequential |       | Laplace |     | for DNNs |     |     |     |     |     |     |
In[RBB18b],theyextendedtheLaplacemethodofSection17.3.2tothesequentialsetting. Specifically,
l e t 1 b e t h e a pp ro xi m ate p os t er i or fr o m th e p re v io us s t e p ; w e as su m e
|     | p (θ 1 :t   | 1 )      | (θ µ | , Λ −t ) |     |     |     |     |     |     |     |
| --- | ----------- | -------- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- |
|     | re|cDis o−n | m≈atNrix | i|s  | t 1 1    |     |     |     |     |     |     |     |
t h e p i K −r o n ec k− e r f a ct o r ed . W e n ow co m p u t e th e n ew m ea n b y so l v i n g t he M A P
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

673
| 17.5. | Online | inference |     |     |     |     |     |     |     |
| ----- | ------ | --------- | --- | --- | --- | --- | --- | --- | --- |
problem
(17.38)
|     | µ =argmaxlogp( |     |     | θ)+logp(θ |     | )     |     |     |     |
| --- | -------------- | --- | --- | --------- | --- | ----- | --- | --- | --- |
|     | t              |     | D   | t |       | |D  | 1:t 1 |     |     |     |
−
1
|     | =argmaxlogp( |     |     | θ)    | (θ µ  | )Λ−t 1 (θ | µ   | )   | (17.39) |
| --- | ------------ | --- | --- | ----- | ----- | --------- | --- | --- | ------- |
|     |              |     | D   | t | − | 2 − t | 1 1       | − t | 1   |         |
|     |              |     |     |       | −     | −         | −   |     |         |
Once we have computed µ , we compute the approximate Hessian at this point, and get the new
t
| posterior |     | precision |     |     |     |     |     |     |     |
| --------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
(17.40)
|     | Λ =λH(µ | )+Λ |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | t       | t   | t 1 |     |     |     |     |     |     |
−
where λ 0 is a weighting factor that trades off how much the model pays attention to the new data
vs old da≥ta.
Now suppose we use a diagonal approximation to the posterior prediction matrix. From Equa-
tion (17.39), we see that this amounts to adding a quadratic penalty to each new MAP estimate, to
encourage it to remain close to the parameters from previous tasks. This approach is called
elastic
| weight |     | consolidation |        | (EWC) | [Kir+17]. |     |      |     |     |
| ------ | --- | ------------- | ------ | ----- | --------- | --- | ---- | --- | --- |
| 17.5.2 |     | Extended      | Kalman |       | filtering | for | DNNs |     |     |
In Section 29.7.2, we showed how Kalman filtering can be used to incrementally compute the
exact posterior for the weights of a linear regression model with known variance, i.e., we compute
| p(θ | ,σ2),   | where   |     | = (u | ,y ):i=1:t | is  | the data | seen so far, and |         |
| --- | ------- | ------- | --- | ---- | ---------- | --- | -------- | ---------------- | ------- |
|     | 1:t     |         | 1:t |      | i i        |     |          |                  |         |
|     | |D      |         | D   | {    |            | }   |          |                  |         |
|     |         | ,θ,σ2)= |     | θTu  | ,σ2)       |     |          |                  | (17.41) |
|     | p(y t u | t       | (y  | t    | t          |     |          |                  |         |
|     | |       |         | N   | |    |            |     |          |                  |         |
is the linear regression likelihood. The application of KF to this model is known as recursive least
squares.
|     | Now consider |         | the case | of nonlinear | regression: |     |     |     |         |
| --- | ------------ | ------- | -------- | ------------ | ----------- | --- | --- | --- | ------- |
|     | p(y u        | ,θ,σ2)= | (y       | f(θ,u        | ),σ2)       |     |     |     | (17.42) |
|     | t            | t       |          | t            | t           |     |     |     |         |
|     | |            |         | N        | |            |             |     |     |     |         |
where is some nonlinear function, such as an MLP. We can use the extended Kalman filter
|     | f(θ,u | t ) |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
(Section 8.3.2) to approximately compute p(θ ,σ2), where θ is the hidden state (see e.g., [SW89;
|     |     |     |     |     |     | t d|Dynamics 1:t |     | t   |     |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --- |
PF03]). To see this, note that we can set the model to the identity function, f(θ )=θ , so
t t
theparametersarepropagatedthroughunchanged,andtheobservationmodeltotheinput-dependent
function f(θ ) = f(θ ,u ). We set the observation noise to R = σ2, and the dynamics noise to
|     |     | t   | t   | t   |     |     |     | t   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
=qI, where is a small constant, to allow the parameters to slowly drift according to artificial
| Q   | t   |     | q   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
noise. (In practice it can be useful to anneal from a large initial value to something near
| process |     |     |     |     |     |     | q   |     |     |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.)
| 17.5.2.1 |     | Example |     |     |     |     |     |     |     |
| -------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
We now give an example of this process in action. We sample a synthetic dataset from the true
function
|     | h (u)=x | 10cos(u)sin(u)+u3 |     |     |     |     |     |     | (17.43) |
| --- | ------- | ----------------- | --- | --- | --- | --- | --- | --- | ------- |
∗
−
| Author: |     | Kevin P. | Murphy. | (C) | MIT Press. | CC-BY-NC-ND |     | license |     |
| ------- | --- | -------- | ------- | --- | ---------- | ----------- | --- | ------- | --- |

674
|     |     |     |         |     |     |     | Chapter 17. | Bayesian | neural | networks |
| --- | --- | --- | ------- | --- | --- | --- | ----------- | -------- | ------ | -------- |
|     |     |     | Step=10 |     |     |     |             | Step=20  |        |          |
|     | 20  |     |         |     |     |     | 20          |          |        |          |
|     |     | 0   |         |     |     |     | 0           |          |        |          |
|     | 20  |     |         |     |     |     | 20          |          |        |          |
|     | −   |     |         |     |     |     | −           |          |        |          |
|     |     | 2   |         | 0   | 2   |     | 2           | 0        | 2      |          |
|     |     | −   |         |     |     |     | −           |          |        |          |
|     |     |     | (a)     |     |     |     |             | (b)      |        |          |
|     |     |     | Step=30 |     |     |     |             | Step=200 |        |          |
|     | 20  |     |         |     |     |     | 20          |          |        |          |
|     |     | 0   |         |     |     |     | 0           |          |        |          |
|     | 20  |     |         |     |     |     | 20          |          |        |          |
|     | −   |     |         |     |     |     | −           |          |        |          |
|     |     | 2   |         | 0   | 2   |     | 2           | 0        | 2      |          |
|     |     | −   |         |     |     |     | −           |          |        |          |
|     |     |     | (c)     |     |     |     |             | (d)      |        |          |
Figure 17.16: Sequential Bayesian inference for the parameters of an MLP using the extended Kalman
filter. We show results after seeing the first 10, 20, 30 and 200 observations. (For a video of this, see
| https://bit.ly/3wXnWaM.) |     |     | Generated |     | by ekf_mlp.ipynb. |     |     |     |     |     |
| ------------------------ | --- | --- | --------- | --- | ----------------- | --- | --- | --- | --- | --- |
and add Gaussian noise with σ =3. We then fit this with an MLP with one hidden layer with H
| hidden | units, | so the model | has | the form |     |     |     |     |     |     |
| ------ | ------ | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- |
(17.44)
| f(θ,u)=W |     | tanh(W | u+b | )+b |     |     |     |     |     |     |
| -------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
|          |     | 2      | 1   | 1   | 2   |     |     |     |     |     |
where W H 1, b H, W 1 H, b 1. We set H =6, so there are D =19 parameters
|           | 1   | R × 1 | R   | 2   | R × | 2 R |     |     |     |     |
| --------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
| in total. | ∈   |       | ∈   | ∈   |     | ∈   |     |     |     |     |
Given the data, we sequentially compute the posterior, starting from a vague Gaussian prior,
), where =100I. (In practice we cannot start from the prior mean, which is
| p(θ)= | (θ 0,Σ | 0   | Σ   | 0   |     |     |     |     |     |     |
| ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
=0,Nsince|
θ linearizing the model around this point results in a zero gradient, so we use an initial
0
random sample for θ .) The results are shown in Figure 17.16. We can see that the model adapts
0
to the data, without having to specify any learning rate. In addition, we see that the predictions
become gradually more confident, as the posterior concentrates on the MLE.
| 17.5.2.2 | Setting | the | variance | terms |     |     |     |     |     |     |
| -------- | ------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- |
In the above example, we set the variance terms by hand. In general we need to estimate the noise
variance σ, which determines and hence the learning rate, as well as the strength of the prior ,
|     |     |     |     | R t |     |     |     |     |     | Σ 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
which controls the amount of regularization. Some methods for doing this are discussed in [FNG00].
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

675
| 17.5.    | Online | inference |     |     |               |     |            |     |     |     |
| -------- | ------ | --------- | --- | --- | ------------- | --- | ---------- | --- | --- | --- |
| 17.5.2.3 |        | Reducing  |     | the | computational |     | complexity |     |     |     |
The naive EKF method described above takes O(N3) time, which is prohibitive for large neural
z
networks. A simple approximation, known as the decoupled EKF, was proposed in [PF91; SPD92]
(see [PF03] for a review). This partitions the weights into groups or blocks, and estimates the
G
relevant matrices for each group g independently. If G=1, this reduces the standard global EKF.
If we put each weight into its own group, we get a fully diagonal approximation. In practice this
does not work any better than SGD, since it ignores correlations between the parameters. A useful
compromise is to put all the weights corresponding to each neuron into its own group; this is called
EKF, which has been used in [Sim02] to train RBF networks and [GUK21] to
node decoupled
train exponential family matrix factorization models (widely used in recommender systems). For
| more | details | on  | DEKF, | Supplementary |     | Section | 17.1. |     |     |     |
| ---- | ------- | --- | ----- | ------------- | --- | ------- | ----- | --- | --- | --- |
Another approach to increasing computational efficiency is to leverage the fact that the effective
dimensionality of a DNN is often quite low (see Section 17.4.3). Indeed we can approximate the
model parameters by using a low dimensional vector of coefficients that specify the point in a linear
manifold corresponding to weight space; the basis set defining this linear manifold can either be
chosen randomly [Li+18b; GARD18; Lar+22], or can be estimated using PCA applied to the SGD
iterates [Izm+19]. We can exploit this observation to perform EKF in this low-dimensional subspace,
| which  | significantly |         | speeds | up      | inference, | as  | discussed | in [DMKM22]. |     |     |
| ------ | ------------- | ------- | ------ | ------- | ---------- | --- | --------- | ------------ | --- | --- |
| 17.5.3 |               | Assumed |        | density | filtering  |     | for DNNs  |              |     |     |
In Section 8.6.3, we discussed how to use assumed density filtering (ADF) to perform online (binary)
logistic regression. In this section, we generalize this to nonlinear predictive models, such as DNNs.
ThekeyistoperformGaussianmomentmatchingofthehiddenactivationsateachlayerofthemodel.
This provides an alternative to the EKF approach in Section 17.5.2, which is based on linearization
of the network.
|     | We will | assume | the | following | likelihood: |     |     |     |     |     |
| --- | ------- | ------ | --- | --------- | ----------- | --- | --- | --- | --- | --- |
(17.45)
|     | p(y u | ,w )=Expfam(y |     |     | ℓ 1(f(u | ;w ))) |     |     |     |     |
| --- | ----- | ------------- | --- | --- | ------- | ------ | --- | --- | --- | --- |
|     | t |   | t t           |     |     | t | −   | t t    |     |     |     |     |
where is the DNN, 1 is the inverse link function, and is some exponential family
|     | f(x;w) |     |     | ℓ   | −   |     |     |     | Expfam() |     |
| --- | ------ | --- | --- | --- | --- | --- | --- | --- | -------- | --- |
distribution. For example, if f is linear and we are solving a binary classification problem, we can
write
|     | p(y u     | ,w )=Ber(y |     | σ(uTw | ))       |       |            |        |     | (17.46) |
| --- | --------- | ---------- | --- | ----- | -------- | ----- | ---------- | ------ | --- | ------- |
|     | t |       | t t        |     | t |   | t t      |       |            |        |     |         |
| We  | discussed | using      | ADF | to    | fit this | model | in Section | 8.6.3. |     |         |
In [HLA15b], they propose probabilistic backpropagation (PBP), which is an instance of ADF
applied to MLPs. The basic idea is to approximate the posterior over the weights in each layer using
| a   | fully factorized |     | distribution |     |       |     |     |     |     |     |
| --- | ---------------- | --- | ------------ | --- | ----- | --- | --- | --- | --- | --- |
|     |                  |     |              |     | Dl Dl | 1+1 |     |     |     |     |
L
|     |       |       |        |                  | −        |     | µt  | ,τt       |     | (17.47) |
| --- | ----- | ----- | ------ | ---------------- | -------- | --- | --- | --------- | --- | ------- |
|     | p(w t | 1:t ) | p t (w | t )=             |          | (w  | ijl | ijl ijl ) |     |         |
|     | |D    | ≈     |        |                  |          | N   | |   |           |     |         |
|     |       |       |        | l=1i=1           | j=1      |     |     |           |     |         |
|     |       |       |        | (cid:89)(cid:89) | (cid:89) |     |     |           |     |         |
where L is the number of layers, and D is the number of neurons in layer l. (The expectation
l
algorithm of [SHM14] is a special case of this, where the variances are fixed to
backpropagation
τ =1.)
| Author: |     | Kevin | P. Murphy. |     | (C) MIT | Press. | CC-BY-NC-ND |     | license |     |
| ------- | --- | ----- | ---------- | --- | ------- | ------ | ----------- | --- | ------- | --- |

676
|     |     |     |     |     |     |     |     | Chapter 17. | Bayesian | neural networks |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | --------------- |
Suppose the parameters are static, so . Then the new posterior, after conditioning on
|          |              |     |     |       |     | w t | =w t 1 |     |     |     |
| -------- | ------------ | --- | --- | ----- | --- | --- | ------ | --- | --- | --- |
| the t’th | observation, |     | is  | given | by  |     | −      |     |     |     |
1
| pˆ  | (w)= | p(y | u ,w) | (w  | µt 1,Σt | 1)  |     |     |     | (17.48) |
| --- | ---- | --- | ----- | --- | ------- | --- | --- | --- | --- | ------- |
| t   |      |     | t t   |     | −       | −   |     |     |     |         |
|     |      | Z   | |     | N   | |       |     |     |     |     |         |
t
where Σt 1 =diag(τt 1). We then project pˆ (w) onto the space of factored Gaussians to compute
|     | −   |     | −   |     |     |     | t   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the new (approximate) posterior, (w). This can be done by computing the following means and
p
t
| variances | [Min01a]: |      |           |     |     |     |     |     |     |         |
| --------- | --------- | ---- | --------- | --- | --- | --- | --- | --- | --- | ------- |
|           |           |      | ∂lnZ      | t   |     |     |     |     |     |         |
| µt        | =µt       | 1+τ  | t 1       |     |     |     |     |     |     | (17.49) |
| ijl       |           | ij−l | i j−l ∂µt | 1   |     |     |     |     |     |         |
ij−l
2
|     |      |       |          | ∂lnZ         |               | ∂lnZ |       |     |     |         |
| --- | ---- | ----- | -------- | ------------ | ------------- | ---- | ----- | --- | --- | ------- |
| τ   | t =τ | t 1   | (τ t 1)2 |              | t             | 2    | t     |     |     | (17.50) |
| i   | jl   | i j−l | i j−l    | (cid:32)∂µt | 1             |      | t 1   |     |     |         |
|     |      | −     |          |              | ij−l (cid:33) | − ∂τ | j−l  |     |     |         |
i
|     |     |     |     |    |     |     |    |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Intheforwardspass,wecomputeZ bypropagatingtheinputu throughthemodel. Sincewehave
|     |     |     |     |     |     | t   |     | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
a Gaussian distribution over the weights, instead of a point estimate, this induces an (approximately)
Gaussian distribution over the values of the hidden units. For certain kinds of activation functions
(suchasReLU),therelevantintegrals(tocomputethemeansandvariances)canbesolvedanalytically,
as in GP-neural networks (Section 18.7). The result is that we get a Gaussian distribution over the
final layer of the form (η µ,Σ), where η =f(u ;w ) is the output of the neural network before
|     |     |     |     | t|ced |     |     | t t | t   |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
the GLM link functionNindu by ). H ence w e ca n approximate the partition function using
|     |     |       |      |        | p t | (w t |     |     |     |         |
| --- | --- | ----- | ---- | ------ | --- | ---- | --- | --- | --- | ------- |
| Z   |     | p(y η | ) (η | µ,Σ)dη |     |      |     |     |     | (17.51) |
| t   |     | t     | t    | t|     | t   |      |     |     |     |         |
|     | ≈   | |     | N    |        |     |      |     |     |     |         |
(cid:90)
Wenowdiscusshowtocomputethisintegral. Inthecaseofprobitclassification,withy 1,+1 ,
wehavep(y x,w)=Φ(yη), whereΦisthecdfofthestandardnormal. Wecanthenuseth∈e{f−ollowin}g
| analytical | re|sult |     |     |     |     |     |     |     |     |     |
| ---------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
yµ
(17.52)
|     | Φ(yη) | (hµ,σ)dη |     | =Φ       |      |          |     |     |     |     |
| --- | ----- | -------- | --- | -------- | ---- | -------- | --- | --- | --- | --- |
|     |       | N |      |     |          | √1+σ |          |     |     |     |     |
|     |       |          |     | (cid:18) |      | (cid:19) |     |     |     |     |
(cid:90)
In the case of logistic classification, with , we have σ(η)); in this case,
|        |     |            |               |     |     | y         | 0,1      | p(y x,w)=Ber(y |             |         |
| ------ | --- | ---------- | ------------- | --- | --- | --------- | -------- | -------------- | ----------- | ------- |
|        |     |            |               |     |     | Se∈ct{ion | 1}5.3.6. | the|multiclass | case,|where |         |
| we can | use | the probit | approximation |     |     | from      |          | For            |             | y 0,1 C |
(one-hot encoding), we have softmax(η)). A variational lower bound to∈ { }for
|           |     |       |               | p(y | x,w)=Cat(y |     |     |     |     | log Z t |
| --------- | --- | ----- | ------------- | --- | ---------- | --- | --- | --- | --- | ------- |
| this case | is  | given | in [GDFY16].| |     |            |     | |   |     |     |         |
Once we have computed Z , we can take gradients and update the Gaussian posterior moments,
t
| before | moving | to  | the next    | step. |           |     |          |     |     |     |
| ------ | ------ | --- | ----------- | ----- | --------- | --- | -------- | --- | --- | --- |
| 17.5.4 | Online |     | variational |       | inference |     | for DNNs |     |     |     |
A natural approach to online learning is to use variational inference, where the prior is the posterior
from the previous step. This is known as streaming variational Bayes [Bro+13]. In more detail,
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

677
| 17.6. Hierarchical |       |         | Bayesian | neural | networks |     |     |     |     |     |
| ------------------ | ----- | ------- | -------- | ------ | -------- | --- | --- | --- | --- | --- |
| at step            | t, we | compute |          |        |          |     |     |     |     |     |
(17.53)
| ψ =arg | minEq(θ |     | [ℓ t | (θ)]+DKL | q(θ | ψ)  | q(θ ψ | )   |     |     |
| ------ | ------- | --- | ---- | -------- | --- | --- | ----- | --- | --- | --- |
| t      |         |     | ψ)   |          |     | | ∥ | |     | t 1 |     |     |
|        | ψ       |     | |    |          |     |     |       | −   |     |     |
Łt(cid:0)(ψ)
(cid:1)
−
=arg minE(cid:124)q(θ ℓ (θ)+logq(cid:123)((cid:122)θ ψ) logq(θ ψ 1(cid:125)) (17.54)
|     |     |     | ψ)      | t   |     |     |     | t       |     |     |
| --- | --- | --- | ------- | --- | --- | --- | --- | ------- | --- | --- |
|     | ψ   |     | |       |     | |   | −   | |   | −       |     |     |
|     |     |     | (cid:2) |     |     |     |     | (cid:3) |     |     |
where is the negative log likelihood (or, more generally, some loss function) of
| ℓ   | (θ)=       | logp( | θ)    |     |     |     |     |     |     |     |
| --- | ---------- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
|     | t batch−at | stepD | t.| t |     |     |     |     |     |     |     |
the data
WhenappliedtoDNNs,thisapproachiscalledvariationalcontinuallearningorVCL[Ngu+18].
(We discuss continual learning in Section 19.7.) An efficient implementation of this, known as FOO-
VB (“fixed-point operator for online variational Bayes”) is given in [Zen+21].
One problem with the VCL objective in Equation (17.53) is that the KL term can cause the
model to become too sparse, which can prevent the model from adapting or learning new tasks.
This problem is called variational overpruning [TT17]. More precisely, the reason this happens
is as follows: some weights might not be needed to fit a given dataset, so their posterior will be
equal to the prior, but sampling from these high-variance weights will add noise to the likeilhood; to
reduce this, the optimization method will prefer to set the bias term to a large negative value, so
the corresponding unit is “turned off”, and thus has no effect on the likelihood. Unfortunately, these
“dead units” become stuck, so there is not enough network capacity to learn the next task.
| In [LST21], |     | they | propose | a   | solution | to this, | known | as          |             |           |
| ----------- | --- | ---- | ------- | --- | -------- | -------- | ----- | ----------- | ----------- | --------- |
|             |     |      |         |     |          |          |       | generalized | variational | continual |
learning or GVCL. The first step is to downweight the KL term by a factor β <1 to get
| Ł       |     |                   |     |     |        |       |       |     |     | (17.55) |
| ------- | --- | ----------------- | --- | --- | ------ | ----- | ----- | --- | --- | ------- |
| t =Eq(θ |     | ψ) [ℓ t (θ)]+βDKL |     |     | q(θ ψ) | q(θ ψ | t 1 ) |     |     |         |
|         | |   |                   |     |     | | ∥    | |     | −     |     |     |         |
Interestingly, one can show tha(cid:0)t in the limit of (cid:1) 0, this recovers several standard methods that
β
Hessia→n.
use a Laplace approximation based on the In particular if we use a diagonal variational
posterior, this reduces to online EWC method of [Sch+18]; if we use a block-diagonal and Kronecker
factored posterior, this reduces to the online structured Laplace method of [RBB18b]; and if we use
a low-rank posterior precision matrix, this reduces to the SOLA method of [Yin+20].
The second step is to replace the prior and posterior by using tempering, which is useful when
the model is misspecified, as discussed in Section 17.3.11. In the case of Gaussians, raising the
distribution to the power is equivalent to tempering with a temperature of =1/λ, which is the
|     |     |     |     | λ   |     |     |     |     | τ   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
same as scaling the covariance by 1. Thus the GVCL objective becomes
λ
−
| Ł =Eq(θ |     | [ℓ (θ)]+βDKL |     |     | q(θ ψ)λ | q(θ ψ | )λ  |     |     | (17.56) |
| ------- | --- | ------------ | --- | --- | ------- | ----- | --- | --- | --- | ------- |
| t       |     | ψ) t         |     |     |         |       | t 1 |     |     |         |
|         | |   |              |     |     | |       | ∥ |   | −   |     |     |         |
This can be optimized using SG(cid:0)D, assuming the post(cid:1)erior is reparameterizable (see Section 10.2.1).
| 17.6 | Hierarchical |     |     | Bayesian | neural |     | networks |     |     |     |
| ---- | ------------ | --- | --- | -------- | ------ | --- | -------- | --- | --- | --- |
In some problems, we have multiple related datasets, such as a set of medical images from different
hospitals. Some aspects of the data (e.g., the shape of healthy vs diseased cells) is generally the same
across datasets, but other aspects may be unique or idiosyncractic (e.g., each hospital may use a
different colored die for staining). To model this, we can use a hierarchical Bayesian model, in which
weallowtheparametersforeachdatasettobedifferent(tocapturerandomeffects),whilecomingfrom
| Author: | Kevin | P. Murphy. |     | (C) | MIT Press. | CC-BY-NC-ND |     | license |     |     |
| ------- | ----- | ---------- | --- | --- | ---------- | ----------- | --- | ------- | --- | --- |

678
|     |     |     |     |     | Chapter 17. | Bayesian | neural | networks |     |
| --- | --- | --- | --- | --- | ----------- | -------- | ------ | -------- | --- |
Twomoonsdataset
|     | 2.0 |     |     |     | Task1 |     | Task2 |     |     |
| --- | --- | --- | --- | --- | ----- | --- | ----- | --- | --- |
1.5
2
1.0
0
0.5
|     | 2X  |     |     | 2   |       |     |        |     |     |
| --- | --- | --- | --- | --- | ----- | --- | ------ | --- | --- |
|     | 0.0 |     |     | −   |       |     |        |     |     |
|     |     |     |     |     | Task3 |     | Task4  |     |     |
|     | 0.5 |     |     |     |       |     | Class0 |     |     |
2
|     | −   |     |     |     |     |     | Class1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- |
1.0
−
0
Class0
1.5
|     | −   |     | Class1 |     |     |     |     |     |     |
| --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
|     | 2.0 |     |        | 2   |     |     |     |     |     |
|     | −   |     |        | −   |     |     |     |     |     |
|     |     | 2   | 1 0 1  | 2 3 | 0.0 | 2.5 | 0.0 | 2.5 |     |
|     | −   | −   |        |     |     |     |     |     |     |
X1
|     |     |     | (a) |     |     | (b) |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Figure 17.17: (a) Two moons synthetic dataset. (b) Multi-task version, where we rotate the data to create
18 related tasks (groups). Each dataset has 50 training and 50 test points. Here we show the first 4 tasks.
| Generated | by  | bnn_hierarchical.ipynb. |     |     |     |     |     |     |     |
| --------- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- |
a common prior (to capture shared effects). This is the setup we considered in Section 15.5, where
we discuss hierarchical Bayesian GLMs. In this section, we extend this to nonlinear predictors based
on neural networks. (The setup is very similar to domain generalization, discussed in Section 19.6.2,
except here we care about performance on all the domains, not just a held-out target domain.)
| 17.6.1 | Example: |     | multimoons | classification |     |     |     |     |     |
| ------ | -------- | --- | ---------- | -------------- | --- | --- | --- | --- | --- |
In this section, we consider an example2 where we want to solve multiple related nonlinear binary
classification problems coming from J different environments or distributions. We assume that each
environment has its own unique decision boundary x,wj), so this is a form of concept shift (see
p(y
Howeverweassumetheoverallshapeof|eachboundaryissimilartoacommonshared
Section19.2.3).
boundary, denote p(y x,w0). We only have a small number N of examples from each environment,
j
| , but we can utilize their common structure to do better than fitting
| j          | = (xj,yj):n= |         | 1:N |     |     |     |     |     | J   |
| ---------- | ------------ | ------- | --- | --- | --- | --- | --- | --- | --- |
| Dsepara{te | n            | n       | j } |     |     |     |     |     |     |
|            | m            | od els. |     |     |     |     |     |     |     |
To illustrate this, we create some synthetic 2d data for the tasks. We start with the
|     |     |     |     |     | J = | 18  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
two-moons dataset, illustrated in Figure 17.17a. Each task is obtained by rotating the 2d inputs by a
different amount, to create 18 related classification problems (see Figure 17.17b). See Figure 17.17b
| for | the training | data | for 4 tasks. |     |     |     |     |     |     |
| --- | ------------ | ---- | ------------ | --- | --- | --- | --- | --- | --- |
To handle the nonlinear decision boundary, we use a multilayer perceptron. Since the dataset is
low-dimensional (2d input), we use a shallow model with just 2 hidden layers, each with 5 neurons.
WecouldfitaseparateMLPtoeachtask, butsincewehavelimiteddatapertask(N =50examples
j
https://twiecki.io/blog/2018/08/13/hierarchical_bayesian_neural_network/.
| 2. This | example | is from |     |     |     |     |     |     | For a |
| ------- | ------- | ------- | --- | --- | --- | --- | --- | --- | ----- |
real-worldexampleofasimilarapproachappliedtoagesturerecognitiontask,see[Jos+17].
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025

679
| 17.6. Hierarchical | Bayesian | neural | networks |     |     |     |     |
| ------------------ | -------- | ------ | -------- | --- | --- | --- | --- |
σ0
3
(cid:15)j
3
|     |     |     | y e |     | w j | w 0 |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | n   |     | 3   | 3   |     |
σ0
|     |     |     |     |     | (cid:15)j | 2   |     |
| --- | --- | --- | --- | --- | --------- | --- | --- |
2
|     |     |     | zj  |     | wj  | w0  |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
2
|     |     |     | 2n  |     | 2   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
σ0
|     |     |     |     |     | (cid:15)j | 1   |     |
| --- | --- | --- | --- | --- | --------- | --- | --- |
1
|     |     |     | zj  |     | wj  | w0  |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | 1n  |     | 1   | 1   |     |
xj
n
n=1:Nj
j=1:J
Figure 17.18: Illustration of a hierarchical Bayesian MLP with 2 hidden layers. There are J different models,
each with observed samples, and a common set of global shared parent parameters denoted with the 0
N j
superscript. Nodeswhichareshadedareobserved. Nodeswithdoubleringedcirclesaredeterministicfunctions
of their parents.
for training), this works poorly, as we show below. We could also pool all the data and fit a single
model, but this does even worse, since the datasets come from different underlying distributions, so
mixingthedatatogetherfromdifferent“concepts” confusesthemodel. Insteadweadoptahierarchical
| Bayesian approach. |     |     |     |     |     |     |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- |
Our modeling assumptions are shown in Figure 17.18. In particular, we assume the weight from
unit to unit in layer for environment j, denoted wj , comes from a common prior value ,
| i   | k   | l   |     |     |       |     | w0    |
| --- | --- | --- | --- | --- | ----- | --- | ----- |
|     |     |     |     |     | i,k,l |     | i,k,l |
with a random offset. We use the non-centered parameterization from Section 12.6.5 to write
| wj =w0 | +ϵj   | σ0       |     |     |     |     | (17.57) |
| ------ | ----- | -------- | --- | --- | --- | --- | ------- |
| i,k,l  | i,k,l | i,k,l× l |     |     |     |     |         |
where ϵj (0,1). By allowing a different σ0 per layer l, we let the model control the degree
| i,k,l | ∼ N |     |     |     | l   |     |     |
| ----- | --- | --- | --- | --- | --- | --- | --- |
of shrinkage to the prior for each layer separately. (We could also make the σj parameters be
l
environment specific, which would allow for different amounts of distribution shift from the common
parent.) For the hyper-parameters, we put (0,1) priors on w0 , and (1) priors on σ0.
|     |     |     |     |     |     | i,k,l + | l   |
| --- | --- | --- | --- | --- | --- | ------- | --- |
|     |     |     |     | N   |     | N       |     |
We compute the posterior p(ϵ1:J,w0 ,σ0 ) using HMC (Section 12.5). We then evaluate this
|     |     |     | 1: L 1 : L | 1 :L|eDachenvironment. |     |     |     |
| --- | --- | --- | ---------- | ---------------------- | --- | --- | --- |
modelusingafreshsetoflabeled s amp l e sfro m Theaverageclassificationaccuracy
on the train and test sets for the non-hierarchical model (one MLP per environment, fit separately)
is 86% and 83%. For the hierarchical model, this improves to 91% and 89% respectively.
To see why the hierarchical model works better, we will plot the posterior predictive distribution
in 2d. Figure 17.19(top) shows the results for the nonhierarchical models; we see that the method
| Author: Kevin | P. Murphy. | (C) | MIT Press. | CC-BY-NC-ND |     | license |     |
| ------------- | ---------- | --- | ---------- | ----------- | --- | ------- | --- |

680
|     |     |         |     |           |     | Chapter | 17. | Bayesian | neural    | networks |
| --- | --- | ------- | --- | --------- | --- | ------- | --- | -------- | --------- | -------- |
|     |     | Dataset | 1   | Dataset 2 |     | Dataset | 3   |          | Dataset 4 |          |
Class0
|     | 2   |     |     |     |     |     |     |     | Class1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
0
2
−
|     | 2.5 | 0.0     | 2.5 | 2.5 0.0 2.5 | 2.5 | 0.0     | 2.5 | 2.5 | 0.0       | 2.5 |
| --- | --- | ------- | --- | ----------- | --- | ------- | --- | --- | --------- | --- |
|     | −   |         | −   |             | −   |         |     | −   |           |     |
|     |     | Dataset | 1   | Dataset 2   |     | Dataset | 3   |     | Dataset 4 |     |
Class0
2
Class1
0
2
−
|     | 2.5 | 0.0 | 2.5 | 2.5 0.0 2.5 | 2.5 | 0.0 | 2.5 | 2.5 | 0.0 | 2.5 |
| --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- |
|     | −   |     | −   |             | −   |     |     | −   |     |     |
Figure 17.19: Top: Results of fitting separate MLPs on each dataset. Bottom: Results of fitting hierarchical
| MLP on | all datasets | jointly. | Generated | by bnn_hierarchical.ipynb. |     |     |     |     |     |     |
| ------ | ------------ | -------- | --------- | -------------------------- | --- | --- | --- | --- | --- | --- |
fails to learn the common underlying Z-shaped decision boundary. By contrast, Figure 17.19(bottom)
shows that the hierarchical method has correctly recovered the common pattern, while still allowing
group variation.
“Probabilistic Machine Learning: Advanced Topics”. Online version. December 10, 2025