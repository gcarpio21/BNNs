PublishedasaconferencepaperatICLR2018
| FLIPOUT:                           |     | EFFICIENT |     | PSEUDO-INDEPENDENT |     |     |     |                       |     | WEIGHT |
| ---------------------------------- | --- | --------- | --- | ------------------ | --- | --- | --- | --------------------- | --- | ------ |
| PERTURBATIONS                      |     |           | ON  | MINI-BATCHES       |     |     |     |                       |     |        |
| YemingWen,PaulVicol,JimmyBa        |     |           |     |                    |     |     |     | DustinTran            |     |        |
| UniversityofToronto                |     |           |     |                    |     |     |     | ColumbiaUniversity    |     |        |
| VectorInstitute                    |     |           |     |                    |     |     |     | Google                |     |        |
| wenyemin,pvicol,jba@cs.toronto.edu |     |           |     |                    |     |     |     | trandustin@google.com |     |        |
RogerGrosse
UniversityofToronto
8102 rpA 2  ]GL.sc[  2v68340.3081:viXra
VectorInstitute
rgrosse@cs.toronto.ca
ABSTRACT
Stochasticneuralnetweightsareusedinavarietyofcontexts,includingregular-
ization,Bayesianneuralnets,explorationinreinforcementlearning,andevolution
|     | strategies.                                 | Unfortunately,duetothelargenumberofweights,alltheexamplesin |       |     |             |                                |     |         |          |     |
| --- | ------------------------------------------- | ----------------------------------------------------------- | ----- | --- | ----------- | ------------------------------ | --- | ------- | -------- | --- |
|     | a mini-batch                                | typically                                                   | share | the | same weight | perturbation,                  |     | thereby | limiting | the |
|     | variancereductioneffectoflargemini-batches. |                                                             |       |     |             | Weintroduceflipout,anefficient |     |         |          |     |
methodfordecorrelatingthegradientswithinamini-batchbyimplicitlysampling
|     | pseudo-independent                               |                 | weight | perturbations |                 | for each    | example.               | Empirically, |           | flipout |
| --- | ------------------------------------------------ | --------------- | ------ | ------------- | --------------- | ----------- | ---------------------- | ------------ | --------- | ------- |
|     | achieves                                         | the ideal       | linear | variance      | reduction       | for         | fully connected        |              | networks, | con-    |
|     | volutional                                       | networks,       | and    | RNNs.         | We find         | significant | speedups               | in           | training  | neural  |
|     | networkswithmultiplicativeGaussianperturbations. |                 |        |               |                 |             | Weshowthatflipoutisef- |              |           |         |
|     | fective                                          | at regularizing | LSTMs, |               | and outperforms |             | previous               | methods.     | Flipout   | also    |
enablesustovectorizeevolutionstrategies:inourexperiments,asingleGPUwith
|     | flipout | can handle | the same | throughput | as  | at least | 40 CPU | cores | using | existing |
| --- | ------- | ---------- | -------- | ---------- | --- | -------- | ------ | ----- | ----- | -------- |
methods,equivalenttoafactor-of-4costreductiononAmazonWebServices.
1 INTRODUCTION
Stochasticityisakeycomponentofmanymodernneuralnetarchitecturesandtrainingalgorithms.
Themostwidelyusedregularizationmethodsarebasedonrandomlyperturbinganetwork’scom-
putations(Srivastavaetal.,2014;Ioffe&Szegedy,2015). Bayesianneuralnetscanbetrainedwith
variationalinferencebyperturbingtheweights(Graves,2011;Blundelletal.,2015). Weightnoise
wasfoundtoaidexplorationinreinforcementlearning(Plappertetal.,2017;Fortunatoetal.,2017).
Evolutionstrategies(ES)minimizesablack-boxobjectivebyevaluatingmanyweightperturbations
inparallel,withimpressiveperformanceonroboticcontroltasks(Salimansetal.,2017).
Some methods perturb a network’s activations (Srivastava et al., 2014; Ioffe & Szegedy, 2015),
whileothersperturbitsweights(Graves,2011;Blundelletal.,2015;Plappertetal.,2017;Fortunato
etal.,2017;Salimansetal.,2017). Stochasticweightsareappealinginthecontextofregularization
orexplorationbecausetheycanbeviewedasaformofposterioruncertaintyabouttheparameters.
However, compared with stochastic activations, they have a serious drawback: because a network
typically has many more weights than units, it is very expensive to compute and store separate
weightperturbationsforeveryexampleinamini-batch. Therefore, stochasticweightmethodsare
typically done with a single sample per mini-batch. In contrast, activations are easy to sample in-
dependentlyfordifferenttrainingexampleswithinamini-batch. Thisallowsthetrainingalgorithm
to see orders of magnitude more perturbations in a given amount of time, and the variance of the
stochastic gradients decays as 1/N, where N is the mini-batch size. We believe this is the main
reason stochastic activations are far more prevalent than stochastic weights for neural net regular-
ization. InothersettingssuchasBayesianneuralnetsandevolutionstrategies,oneisforcedtouse
weightperturbationsandlivewiththeresultinginefficiency.
1

PublishedasaconferencepaperatICLR2018
In order to achieve the ideal 1/N variance reduction, the gradients within a mini-batch need not
be independent, but merely uncorrelated. In this paper, we present flipout, an efficient method
for decorrelating the gradients between different examples without biasing the gradient estimates.
Flipout applies to any perturbation distribution that factorizes by weight and is symmetric around
0—including DropConnect, multiplicative Gaussian perturbations, evolution strategies, and varia-
tional Bayesian neural nets—and to many architectures, including fully connected nets, convolu-
tionalnets,andRNNs.
In Section 3, we show that flipout gives unbiased stochastic gradients, and discuss its efficient
vectorizedimplementationwhichincursonlyafactor-of-2computationaloverheadcomparedwith
sharedperturbations.Wethenanalyzetheasymptoticsofgradientvariancewithandwithoutflipout,
demonstratingstrictlyreducedvariance. InSection4,wemeasurethevariancereductioneffectson
a variety of architectures. Empirically, flipout gives the ideal 1/N variance reduction in all archi-
tectures we have investigated, just as if the perturbations were done fully independently for each
training example. We demonstratespeedups in trainingtime in a largebatch regime. We also use
flipout to regularize the recurrent connections in LSTMs, and show that it outperforms methods
based on dropout. Finally, we use flipout to vectorize evolution strategies (Salimans et al., 2017),
allowingasingleGPUtohandlethesamethroughputas40CPUcoresusingexistingapproaches;
thiscorrespondstoafactor-of-4costreductiononAmazonWebServices.
2 BACKGROUND
2.1 WEIGHTPERTURBATIONS
Weusetheterm“weightperturbation”torefertoaclassofmethodswhichsampletheweightsof
aneuralnetworkstochasticallyattrainingtime. Moreprecisely,letf(x,W)denotetheoutputofa
networkwithweightsW oninputx. Theweightsaresampledfromadistributionq parameterized
θ
by θ. We aim to minimize the expected loss E [L(f(x,W),y)], where L is a loss
(x,y)∼D,W∼qθ
function, and D denotes the data distribution. The distribution q can often be described in terms
θ
ofperturbations: W =W +∆W,whereW arethemeanweights(typicallyrepresentedexplicitly
aspartofθ)and∆W isastochasticperturbation. Wenowgivesomespecificexamplesofweight
perturbations.
Gaussianperturbations. Iftheentries∆W aresampledindependentlyfromGaussiandistribu-
ij
tions with variance σ2, this corresponds to the distribution W ∼ N(W ,σ2). Using the repa-
ij ij ij ij
rameterizationtrick(Kingma&Welling,2014),thiscanberewrittenasW =W +σ (cid:15) ,where
ij ij ij ij
(cid:15) ∼ N(0,1); thisrepresentationallowsthegradientstobecomputedusingbackprop. Avariant
ij
ofthisismultiplicativeGaussianperturbation, wheretheperturbationsarescaledaccordingtothe
weights: W ∼ N(W ,σ2W 2 ),orW = W (1+σ (cid:15) ),whereagain(cid:15) ∼ N(0,1). Multi-
ij ij ij ij ij ij ij ij ij
plicativeperturbationscanbemoreeffectivethanadditiveonesbecausetheinformationcontentof
theweightsisthesameregardlessoftheirscale.
DropConnect. DropConnect (Wan et al., 2013) is a regularization method inspired by dropout
(Srivastava et al., 2014) which randomly zeros out a random subset of the weights. In the case of
a50%droprate, thiscanbethoughtofasaweightperturbationwhereW = W/2andeachentry
∆W issampleduniformlyfrom±W .
ij ij
VariationalBayesianneuralnets.Ratherthanfittingapointestimateofaneuralnet’sweights,one
canadopttheBayesianapproachofputtingapriordistributionp(W)overtheweightsandapprox-
imating the posterior distribution p(W|D) ∝ p(W)p(D|W), where D denotes the observed data.
Graves (2011) observed that one could fit an approximation q (W) ≈ p(W|D) using variational
θ
inference;inparticular,onecouldmaximizetheevidencelowerbound(ELBO)withrespecttoθ:
F(θ)= E [logp(D|W)]−D (q (cid:107)p).
KL θ
W∼qθ
Thenegationofthesecondtermcanbeviewedasthedescriptionlengthofthedata,andthenegation
ofthefirsttermcanbeviewedasthedescriptionlengthoftheweights(Hinton&VanCamp,1993).
Graves(2011)observedthatifqischosentobeafactorialGaussian,samplingfromθcanbethought
of as Gaussian weight perturbation where the variance is adapted to maximize F. Blundell et al.
2

PublishedasaconferencepaperatICLR2018
(2015) later combined this insight with the reparameterization trick (Kingma & Welling, 2014) to
deriveunbiasedstochasticestimatesofthegradientofF.
Evolutionstrategies. ES(Rechenberg&Eigen,1973)isafamilyofblackboxoptimizationalgo-
rithmswhichuseweightperturbationstosearchformodelparameters. ESwasrecentlyproposedas
an alternative reinforcement learning algorithm (Schmidhuber et al., 2007; Salimans et al., 2017).
Ineachiteration,ESgeneratesacollectionofweightperturbationsascandidatesandevaluateseach
accordingtoafitnessfunctionF. Thegradientoftheparameterscanbeestimatedfromthefitness
functionevaluations. ESishighlyparallelizable,becauseperturbationscanbegeneratedandeval-
|     |     |     |     |     |     | M   |     |     | W   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
uated independently by different workers. Suppose is the number of workers, is the model
parameter,σ isthestandarddeviationoftheperturbations,αisthelearningrate,F istheobjective
function,and∆W istheGaussiannoisegeneratedatworkerm. TheESalgorithmtriestomax-
|     | (cid:2) (cid:0) | m   | (cid:1)(cid:3) |     |     |     |     |     |     |     |
| --- | --------------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
imize E F W +σ∆W . The gradient of the objective function and the update rule can be
∆W
givenas:
|     | (cid:2) |      | (cid:3) | 1   | (cid:2) |          | (cid:3) |         |          |     |
| --- | ------- | ---- | ------- | --- | ------- | -------- | ------- | ------- | -------- | --- |
| ∇   | E F(W   | +∆W) | =       | E   | ∆WF(W   | +∆W)     | ,       | where∆W | ∼N(0,σI) |     |
| W   |         |      |         | σ2  |         |          |         |         |          |     |
|     | ∆W      |      |         | ∆W  |         |          |         |         |          |     |
|     |         |      |         |     |         | M        |         |         |          | (1) |
|     |         |      |         |     | 1       | (cid:88) |         |         |          |     |
|     |         | =⇒ W | =W      | +α  |         | F(W      | +∆W     | )∆W     |          |     |
|     |         |      | t+1     | t   |         |          | t       | m       | m        |     |
Mσ2
m=1
2.2 LOCALREPARAMETERIZATIONTRICK
Insomecases,it’spossibletoreformulateweightperturbationsasactivationperturbations,thereby
allowingthemtobeefficientlycomputedfullyindependentlyfordifferentexamplesinamini-batch.
Inparticular,Kingmaetal.(2015)showedthatforfullyconnectednetworkswithnoweightsharing,
unbiased stochastic gradients could be computed without explicit weight perturbations using the
local reparameterization trick (LRT). For example, suppose X is the input mini-batch, W is the
weightmatrixandB =XW isthematrixofactivations. TheLRTsamplestheactivationsBrather
| thantheweightsW. |       | InthecaseofaGaussianposterior,theLRTisgivenby: |         |      |     |          |         |         |         |     |
| ---------------- | ----- | ---------------------------------------------- | ------- | ---- | --- | -------- | ------- | ------- | ------- | --- |
|                  | q (W  | )=N(µ                                          | ,σ2     | ) ∀W | ∈W  | =⇒       | q (b    | |X)=N(γ | ,δ )    |     |
|                  | θ i,j |                                                | i,j i,j |      | i,j |          | θ m,j   |         | m,j m,j |     |
|                  |       | (cid:88)                                       |         |      |     | (cid:88) |         |         |         |     |
|                  | γ     | =                                              | x µ     | ,    | δ   | =        | x2 σ2   | ,       |         | (2) |
|                  | m,j   |                                                | m,i i,j | and  | m,j |          | m,i i,j |         |         |     |
|                  |       | i=1                                            |         |      |     | i=1      |         |         |         |     |
whereb denotestheperturbedactivations. WhiletheexactLRTappliesonlytofullyconnected
m,j
networks with no weight sharing, Kingma et al. (2015) also introduced variational dropout, a reg-
ularizationmethodinspiredbytheLRTwhichperformswellempiricallyevenforarchitecturesthe
LRTdoesnotapplyto.
2.3 OTHERRELATEDWORK
Control variates are another general class of strategies for variance reduction, both for black-box
optimization(Williams,1992;Ranganathetal.,2014;Mnih&Gregor,2014)andforgradient-based
optimization (Roeder et al., 2016; Miller et al., 2017; Louizos et al., 2017). Control variates are
complementary to flipout, so one could potentially combine these techniques to achieve a larger
variance reduction. We also note that the fastfood transform (Le et al., 2013) is based on similar
mathematicaltechniques. However,whereasfastfoodisusedtoapproximatelymultiplybyalarge
Gaussian matrix, flipout preserves the random matrix’s distribution and instead decorrelates the
gradientsbetweendifferentsamples.
3 METHODS
Asdescribedabove, weightperturbationalgorithmssufferfromhighvarianceofthegradientesti-
mates because all training examples in a mini-batch share the same perturbation. More precisely,
sharingtheperturbationinducescorrelationsbetweenthegradients,implyingthatthevariancecan’t
be eliminated by averaging. In this section, we introduce flipout, an efficient way to perturb the
weightsquasi-independentlywithinamini-batch.
3

PublishedasaconferencepaperatICLR2018
3.1 FLIPOUT
Wemaketwoassumptionsabouttheweightdistributionq :(1)theperturbationsofdifferentweights
θ
areindependent;and(2)theperturbationdistributionissymmetricaroundzero.Thesearenontrivial
constraints, but they encompass important use cases: independent Gaussian perturbations (e.g. as
used in variational BNNs and ES) and DropConnect with drop probability 0.5. We observe that,
undertheseassumptions,theperturbationdistributionisinvarianttoelementwisemultiplicationby
arandomsignmatrix(i.e.amatrixwhoseentriesare±1). Inthefollowing,wedenoteelementwise
multiplicationby◦.
Observation 1. Let q be a perturbation distribution that satisfies the above assumptions, and let
θ
∆(cid:100)W ∼ q
θ
. LetE bearandomsignmatrixthatisindependentof∆(cid:100)W. Then∆W = ∆(cid:100)W ◦E is
identicallydistributedto∆(cid:100)W. Furthermore,thelossgradientscomputedusing∆W areidentically
distributedtothosecomputedusing∆(cid:100)W.
Flipoutexploitsthisfactbyusingabaseperturbation∆(cid:100)W sharedbyallexamplesinthemini-batch,
andmultipliesitbyadifferentrank-onesignmatrixforeachexample:
∆W
n
=∆(cid:100)W ◦r
n
s(cid:62)
n
, (3)
where the subscript denotes the index within the mini-batch, and r and s are random vectors
n n
whoseentriesaresampleduniformlyfrom±1. AccordingtoObservation1,themarginaldistribu-
tion over gradients computed for individual training examples will be identical to the distribution
computedusingsharedweightperturbations. Consequently,flipoutyieldsanunbiasedestimatorfor
thelossgradients. However,bydecorrelatingthegradientsbetweendifferenttrainingexamples,we
canachievemuchlowervarianceupdateswhenaveragingoveramini-batch.
Vectorization. Theadvantageofflipoutoverexplicitperturbationsisthatcomputationsonamini-
batchcanbewrittenintermsofmatrixmultiplications. Thisenablesefficientimplementationson
GPUsandmodernacceleratorssuchastheTensorProcessingUnit(TPU)(Jouppietal.,2017). Let
xdenotetheactivationsinonelayerofaneuralnet. Thenextlayer’sactivationsaregivenby:
y =φ (cid:0) W(cid:62)x (cid:1)
n n
(cid:18)(cid:16) (cid:17)(cid:62) (cid:19)
=φ W +∆(cid:100)W ◦r
n
s(cid:62)
n
x
n
(cid:16) (cid:62) (cid:16) (cid:62) (cid:17) (cid:17)
=φ W x
n
+ ∆(cid:100)W (x
n
◦s
n
) ◦r
n
,
whereφdenotestheactivationfunction. Tovectorizethesecomputations,wedefinematricesRand
S whoserowscorrespondtotherandomsignvectorsr ands forallexamplesinthemini-batch.
n n
Theaboveequationisvectorizedas:
(cid:16) (cid:16) (cid:17) (cid:17)
Y =φ XW + (X◦S)∆(cid:100)W ◦R . (4)
Thisdefinestheforwardpass. BecauseRandS aresampledindependentlyofW and∆(cid:100)W,wecan
backpropagatethroughEqn.4toobtainderivativeswithrespecttoW,∆(cid:100)W,andX.
Computational cost. In general, the most expensive operation in the forward pass is matrix mul-
tiplication. Flipout’sforwardpassrequirestwomatrixmultiplicationsinsteadofone,andtherefore
shouldberoughlytwiceasexpensiveasaforwardpasswithasinglesharedperturbationwhenthe
multiplicationsaredoneinsequence.1 However,notethatthetwomatrixmultiplicationsareinde-
pendentandcanbedoneinparallel; thisincursthesameoverheadasthelocalreparameterization
trick(Kingmaetal.,2015).
A general rule of thumb for neural nets is that the backward pass requires roughly twice as many
FLOPs as the forward pass. This suggests that each update using flipout ought to be about twice
asexpensiveasanupdatewithasinglesharedperturbation(ifthematrixmultiplicationsaredone
sequentially);thisisconsistentwithourexperience.
1Dependingontheefficiencyoftheunderlyinglibraries,theoverheadofsamplingRandS maybenon-
negligible. Ifthisisanissue,thesematricesmaybereusedbetweenallmini-batches. Inourexperience,this
doesnotcauseanydropinperformance.
4

PublishedasaconferencepaperatICLR2018
Evolution strategies. ES is a highly parallelizable algorithm; however, most ES systems are en-
gineered to run on multi-core CPU machines and are not able to take full advantage of GPU par-
allelism. Flipout enables ES to run more efficiently on a GPU because it allows each worker to
evaluateabatchofquasi-independentperturbationsratherthanonlyasingleperturbation. Toapply
flipouttoES,wecansimplyreplicatethestartingstatebythenumberofflipoutperturbationsN,at
| eachworker. | InsteadofEqn.1,theupdateruleusingM |     |     |     |            |            | workersbecomes: |     |     |           |     |
| ----------- | ---------------------------------- | --- | --- | --- | ---------- | ---------- | --------------- | --- | --- | --------- | --- |
|             |                                    |     |     | 1   | (cid:88) M | (cid:88) N | (cid:110)       |     |     | (cid:111) |     |
s(cid:62)
|     |     | W t+1 | =W t | +α   |     | F mn | ∆(cid:100)W | m ◦r | mn  |     | (5) |
| --- | --- | ----- | ---- | ---- | --- | ---- | ----------- | ---- | --- | --- | --- |
|     |     |       |      | MNσ2 |     |      |             |      | mn  |     |     |
m=1n=1
where m indexes workers, n indexes the examples in a worker’s batch, and F is the reward
mn
evaluated with the nth perturbation at worker m. Hence, each worker is able to evaluate multiple
perturbationsasabatch,allowingforparallelismonaGPUarchitecture.
3.2 VARIANCEANALYSIS
Inthissection, weanalyzethevarianceofstochasticgradientswithandwithoutflipout. Weshow
thatflipoutisguaranteedtoreducethevarianceofthegradientestimatescomparedtousingna¨ıve
sharedperturbations.
Let G = G(x,∆W) = ∂ L(y,f(x,W,∆W)) denote one entry of the stochastic gradient
| x   |     |     | ∂θi |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∇ L(y,f(x,W,∆W)) under the perturbation ∆W for a single training example x. (Note that
θ
G isarandomvariablewhichdependsonbothxand∆W. Weanalyzeasingleentryofthegra-
x
dient so that we can work with scalar-valued variances.) We denote the gradient averaged over a
(cid:80)N
mini-batch as the random variable G = 1 G(x ,∆W ), where B = {x }N denotes a
|     |     |     |     | B   | N n=1 |     | n   | n   |     | n n=1 |     |
| --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | ----- | --- |
nth
mini-batch of size N, and ∆W denotes the perturbation for the example. (The randomness
n
comesfromboththechoiceofBandtherandomperturbations.) Forsimplicity,weassumethatthe
x aresampledi.i.d.fromthedatadistribution.
n
UsingtheLawofTotalVariance,wedecomposeVar(G )intoadataterm(thevarianceoftheexact
B
mini-batchgradients)andanestimationterm(theestimationvarianceforafixedmini-batch):
|     |     |       |       | (cid:18)  |                    | (cid:19)  | (cid:104) |                    | (cid:1)(cid:105) |     |     |
| --- | --- | ----- | ----- | --------- | ------------------ | --------- | --------- | ------------------ | ---------------- | --- | --- |
|     |     |       |       |           | (cid:2)            | (cid:3)   |           | (cid:0)            |                  |     |     |
|     |     | Var(G | )=Var |           | E G                | |B +E     | Var       | G |B               | .                |     | (6) |
|     |     |       | B     |           | B                  |           |           | B                  |                  |     |     |
|     |     |       |       | B         | ∆W                 |           | B ∆W      |                    |                  |     |     |
|     |     |       |       | (cid:124) | (cid:123)(cid:122) | (cid:125) | (cid:124) | (cid:123)(cid:122) | (cid:125)        |     |     |
|     |     |       |       |           | data               |           |           | estimation         |                  |     |     |
NoticethatthedatatermdecayswithN whiletheestimationtermmaynot,duetoitsdependence
onthesharedperturbation. Butwecanbreaktheestimationtermintotwopartsforwhichwecan
analyzethedependenceonN. Todothis,wereformulatethestandardsharedperturbationscheme
asfollows:∆W isgeneratedbyfirstsampling∆(cid:100)W andthenmultiplyingitbyarandomsignmatrix
rs(cid:62) as in Eqn. 3 — exactly like flipout, except that the sign matrix is shared by the whole mini-
batch. According to Observation 1, this yields an identical distribution for ∆W to the standard
| sharedperturbationscheme.               |     |       | Basedonthis,weobtainthefollowingdecomposition: |                 |                              |                |           |      |     |     |     |
| --------------------------------------- | --- | ----- | ---------------------------------------------- | --------------- | ---------------------------- | -------------- | --------- | ---- | --- | --- | --- |
| Theorem2(VarianceDecompositionTheorem). |     |       |                                                |                 |                              | Defineα,β,andγ |           | tobe |     |     |     |
|                                         |     |       | (cid:18)                                       |                 | (cid:19)                     |                |           |      |     |     |     |
|                                         |     |       |                                                | (cid:2) (cid:3) | (cid:104)                    |                | (cid:105) |      |     |     |     |
|                                         |     | α=Var | E                                              | G |x            | +E                           | Var(G          | |x)       |      |     |     | (7) |
|                                         |     |       |                                                | x               |                              | x              |           |      |     |     |     |
|                                         |     |       | x ∆W                                           |                 | x                            | ∆W             |           |      |     |     |     |
|                                         |     |       |                                                | (cid:104)       |                              |                | (cid:105) |      |     |     |     |
|                                         |     | β =   | E                                              | Cov(G           | ,G |x,x(cid:48),∆(cid:100)W) |                |           |      |     |     | (8) |
x x(cid:48)
|     |     | x,x(cid:48),∆(cid:100)W |             | ∆W       |                  |     |                                        |     |     |                  |     |
| --- | --- | ----------------------- | ----------- | -------- | ---------------- | --- | -------------------------------------- | --- | --- | ---------------- | --- |
|     |     |                         | (cid:20)    | (cid:18) |                  |     |                                        |     |     | (cid:19)(cid:21) |     |
|     |     | γ =                     | E Cov       | E [G     | |x,∆(cid:100)W], | E   | [G |x(cid:48),∆(cid:100)W]|x,x(cid:48) |     |     |                  | (9) |
|     |     |                         |             |          | x                |     | x(cid:48)                              |     |     |                  |     |
|     |     | x,x(cid:48)             | ∆(cid:100)W | ∆W       |                  | ∆W  |                                        |     |     |                  |     |
UndertheassumptionsofObservation1,thevarianceofthegradientsundersharedperturbations
| andflipoutperturbationscanbewrittenintermsofα,β,andγ |     |     |     |     |     |     |     | asfollows: |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- |
1
|     | Fullyindependentperturbations:Var(G |     |     |     |     |     | )=  | α   |     |     | (10) |
| --- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
B
N
|     |     |     |                     |          |     |       | 1   | N   | −1    |     |      |
| --- | --- | --- | ------------------- | -------- | --- | ----- | --- | --- | ----- | --- | ---- |
|     |     |     | Sharedperturbation: |          |     | Var(G | )=  | α+  | (β+γ) |     | (11) |
|     |     |     |                     |          |     | B     | N   |     | N     |     |      |
|     |     |     |                     |          |     |       | 1   | N   | −1    |     |      |
|     |     |     |                     | Flipout: |     | Var(G | )=  | α+  | γ     |     | (12) |
|     |     |     |                     |          |     | B     | N   |     | N     |     |      |
5

PublishedasaconferencepaperatICLR2018
Proof. DetailsoftheproofareprovidedinAppendixA.
We can interpret α, β, and γ as follows. First, α combines the data term from Eqn. 6 with the
expectedestimationvarianceforindividualdatapoints. Thiscorrespondstothevarianceofthegra-
dientsonindividualtrainingexamples, sofullyindependentperturbationsyieldatotalvarianceof
α/N. The other terms, β and γ, reflect the covariance between the estimation errors on different
training examples as a result of the shared perturbations. The term β reflects the covariance that
resultsfromsamplingr ands,soitiseliminatedbyflipout,whichsamplesthesevectorsindepen-
dently. Finally, γ reflects the covariance that results from sampling ∆(cid:100)W, which flipout does not
eliminate.
Empirically,foralltheneuralnetworksweinvestigated,wefoundthatα (cid:29) β (cid:29) γ. Thisimplies
thefollowingbehaviorforVar(G )asafunctionofN: forsmallN,thedatatermα/N dominates,
B
givinga1/N variancereduction;withsharedperturbations,onceN islargeenoughthatα/N <β,
thevarianceVar(G )levelsofftoβ. However,flipoutcontinuestoenjoya1/N variancereduction
B
inthisregime. Inprinciple,flipout’svarianceshouldleveloffatthepointwhereα/N < γ,butin
allofourexperiments,γ wassmallenoughthatthisneveroccurred: flipout’svariancewasapproxi-
matelyα/N throughouttheentirerangeofN valuesweexplored,justasiftheperturbationswere
sampledfullyindependentlyforeverytrainingexample.
4 EXPERIMENTS
We first verified empirically the variance reduction effect of flipout predicted by Theorem 2; we
measured the variance of the gradients under different perturbations for a wide variety of neural
network architectures and batch sizes. In Section 4.2, we show that flipout applied to Gaussian
perturbations and DropConnect is effective at regularizing LSTM networks. In Section 4.3, we
demonstrate that flipout converges faster than shared perturbations when training with large mini-
batches. Finally,inSection4.4wepresentexperimentscombiningEvolutionStrategieswithflipout
inbothsupervisedlearningandreinforcementlearningtasks.
Inourexperiments,weconsiderthefourarchitecturesshowninTable1(detailsinAppendixB).
4.1 VARIANCEREDUCTION
Since the main effect of flipout is intended to be variance reduction of the gradients, we first esti-
matedthegradientvariancesofseveralarchitectureswithmini-batchsizesrangingfrom1to8196
(Fig.1). Weexperimentedwiththreeperturbationmethods: asinglesharedperturbationpermini-
batch,thelocalreparameterizationtrick(LRT)ofKingmaetal.(2015),andflipout.
ForeachoftheFC,ConVGG,andLSTMarchitectures,wefrozeapartiallytrainednetworktouse
for all variance estimates, and we used multiplicative Gaussian perturbations with σ2 = 1. We
computed Monte Carlo estimates of the gradient variance, including both the data and estimation
termsinEqn.6. Confidenceintervalsarebasedon50independentrunsoftheestimator. Detailsare
giveninAppendixC.
The analysis in Section 3.2 makes strong predictions about the shapes of the curves in Fig. 1. By
Theorem 2, the variance curves for flipout and shared perturbations each have the form a+b/N,
whereN isthemini-batchsize. Onalog-logplot, thisfunctionalformappearsasalinearregime
withslope-1,aconstantregime,andasmoothphasetransitioninbetween. Also,becausethedistri-
butionofindividualgradientsisidenticalwithandwithoutflipout,thecurvesmustagreeforN =1.
Name NetworkType DataSet
ConvLe (Shallow)Convolutional MNIST(LeCunetal.,1998)
ConVGG (Deep)Convolutional CIFAR-10(Krizhevsky&Hinton,2009)
FC FullyConnected MNIST
LSTM LSTMNetwork PennTreebank(Marcusetal.,1993)
Table1: NetworkConfigurations
6

PublishedasaconferencepaperatICLR2018
Variance Estimation
|     | Variance Estimation |     | Variance Estimation |     |     |     |     |
| --- | ------------------- | --- | ------------------- | --- | --- | --- | --- |
101
102
| 102 |     |     |     |     | 108 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
103
| 103      |     |          |     |     | ecnairaV |     |     |
| -------- | --- | -------- | --- | --- | -------- | --- | --- |
| ecnairav |     | ecnairav |     |     | 109      |     |     |
| 104      |     | 104      |     |     |          |     |     |
Wf
| 105 |     | 105 |     |     | 1010 |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- |
Wi
| 106                       |                 | 106                         |             |     | Wo   |            |         |
| ------------------------- | --------------- | --------------------------- | ----------- | --- | ---- | ---------- | ------- |
|                           | FC1             |                             | Conv1       |     | 1011 |            |         |
| 107                       |                 |                             |             |     | Wc   |            |         |
|                           | FC3             | 107                         | Conv8       |     |      |            |         |
| 108                       |                 |                             |             |     | 101  | 102        | 103 104 |
|                           | 100 101 102 103 | 104                         | 100 101 102 | 103 | 104  | Batch Size |         |
|                           | Batch size      |                             | Batch size  |     |      |            |         |
| (a)Fully-connectedNet(FC) |                 | (b)ConvolutionalNet(conVGG) |             |     |      | (c)LSTM    |         |
Figure 1:
Empirical variance of gradients with respect to mini-batch size for several architectures. (a) FC
on MNIST; FC1 denotes the first layer of the FC network. (b) ConVGG on CIFAR-10; Conv1 denotes the
firstconvolutionallayer. (c)LSTMonPennTreebank;thevarianceisshownforthehidden-to-hiddenweight
matricesinthefirstLSTMlayer: W ,W ,W ,andW aretheweightsfortheforget,inputandoutputgates,
|     |     | f i | o c |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
andthecandidatecellupdate,respectively.Dotted:sharedperturbations.Solid:flipout.Dashed:LRT.
Our plots are consistent with both of these predictions. We observe that for shared perturbations,
thephasetransitionconsistentlyoccursformini-batchsizessomewherebetween100and1000. In
contrast, flipout gives the ideal linear variance reduction throughout the range of mini-batch sizes
weinvestigated,i.e.,itsbehaviorisindistinguishablefromfullyindependentperturbations.
AsanalyzedbyKingmaetal.(2015),theLRTgradientsarefullyindependentwithinamini-batch,
andarethereforeguaranteedtoachievetheideal1/N variancereduction. Furthermore,theyreduce
thevariancebelowthatofexplicitweightperturbations,sowewouldexpectthemtoachievesmaller
variance than flipout, as shown in Fig. 1a. However, flipout is applicable to a wider variety of
architectures,includingconvolutionalnetsandRNNs.
4.2 REGULARIZATIONFORLANGUAGEMODELING
We evaluated the regularization effect of flipout on the character-level and word-level language
modelingtaskswiththePennTreebankcorpus(PTB)(Marcusetal.,1993). Wecomparedflipout
to severalother methods forregularizing RNNs: na¨ıve dropout(Zaremba et al., 2014), variational
dropout (Gal & Ghahramani, 2016), recurrent dropout (Semeniuta et al., 2016), zoneout (Krueger
et al., 2016), and DropConnect (Merity et al., 2017). Zaremba et al. (2014)apply dropout only to
thefeed-forwardconnectionsofanRNN(totheinput,output,andconnectionsbetweenlayers).The
othermethodsregularizetherecurrentconnectionsaswell: Semeniutaetal.(2016)applydropout
to the cell update vector, with masks sampled either per step or per sequence; Gal & Ghahramani
(2016)applydropouttotheforwardandrecurrentconnections,withalldropoutmaskssampledper
sequence. Merityetal.(2017)useDropConnecttoregularizethehidden-to-hiddenweightmatrices,
withasingleDropConnectmasksharedbetweenexamplesinamini-batch. Wedenotetheirmodel
WD(forweight-droppedLSTM).
Character-Level. For our character-level experiments, we used a single-layer LSTM with 1000
hidden units (4.26M total parameters). We trained each model on non-overlapping sequences of
100 characters in batches of size 32, using the AMSGrad variant of Adam (Reddi et al., 2018)
with learning rate 0.002. We perform early stopping based on validation performance. Here, we
applied flipout to the hidden-to-hidden weight matrix. More hyperparameter details are given in
AppendixD.Theresults,measuredinbits-per-character(BPC)forthevalidationandtestsequences
of PTB, are shown in Table 2. In the table, shared perturbations and flipout (with Gaussian noise
sampling)aredenotedbyMult. GaussandMult. Gauss+Flipout, respectively. Wealsocompare
to RBN (recurrent batchnorm) (Cooijmans et al., 2017) and H-LSTM+LN (HyperLSTM + Layer-
Norm) (Ha et al., 2016). Mult. Gauss + Flipout outperforms the other methods, and achieves the
bestreportedresultsforthisarchitecture.
Word-Level. Forourword-levelexperiments,weuseda2-layerLSTMwith650hiddenunitsper
layer and 650-dimensional word embeddings (19.8M total parameters). We trained on sequences
oflength35inbatchesofsize40,for100epochs. WeusedSGDwithinitiallearningrate30,and
decayedthelearningratebyafactorof4basedonthenonmonotoniccriterionintroducedbyMerity
et al. (2017). We used flipout to implement DropConnect, as described in Section 2.1, and call
this WD+Flipout. We applied WD+Flipout to the hidden-to-hidden weight matrices for recurrent
regularization, and used the same hyperparameters as Merity et al. (2017). We used embedding
7

PublishedasaconferencepaperatICLR2018
|                   | Model     | Valid Test  |                   |        |        |
| ----------------- | --------- | ----------- | ----------------- | ------ | ------ |
| UnregularizedLSTM |           | 1.468 1.423 | Model             | Valid  | Test   |
| Semeniuta(2016)   |           | 1.337 1.300 | UnregularizedLSTM | 132.23 | 128.97 |
|                   |           |             | Zaremba(2014)     | 80.40  | 76.81  |
| Zoneout(2016)     |           | 1.306 1.270 |                   |        |        |
|                   |           |             | Semeniuta(2016)   | 81.91  | 77.88  |
| Gal(2016)         |           | 1.277 1.245 |                   |        |        |
|                   |           |             | Gal(2016)         | 78.24  | 75.39  |
| Mult.Gauss(σ      | =1)(ours) | 1.257 1.230 |                   |        |        |
Mult.Gauss+Flipout(ours) 1.256 1.227 Zoneout(2016) 78.66 75.45
| RBN(2017)       |     | – 1.32      | WD(2017)            | 78.82                 | 75.71   |
| --------------- | --- | ----------- | ------------------- | --------------------- | ------- |
| H-LSTM+LN(2016) |     | 1.281 1.250 | WD+Flipout(ours)    | 76.88                 | 73.20   |
|                 |     |             | Table 3: Perplexity | on the PTB word-level | valida- |
Table2: Bits-per-character(BPC)forthecharacter-
tionandtestsets.Allresultsarefromourownexper-
| levelPTBtask. | TheRBNandH-LSTM+LNresults |     |     |     |     |
| ------------- | ------------------------- | --- | --- | --- | --- |
iments.
| arefromtherespectivepapers. |     | Allotherresultsare |     |     |     |
| --------------------------- | --- | ------------------ | --- | --- | --- |
fromourownexperiments.
dropout(settingrowsoftheembeddingmatrixto0)withprobability0.1forallregularizedmodels
exceptGal,whereweusedprobability0.2asspecifiedintheirpaper. Morehyperparameterdetails
aregiveninAppendixD.WeshowinTable3thatWD+Flipoutoutperformstheothermethodswith
respect to both validation and test perplexity. In Appendix E.4, we show that WD+Flipout yields
significant variance reduction for large mini-batches, and that when training with batches of size
8192,itconvergesfasterthanWD.
4.3 LARGEBATCHTRAININGWITHFLIPOUT
Theorem2andFig.1suggestthatthevariancereductioneffectofflipoutismorepronouncedinthe
large mini-batch regime. In this section, we train a Bayesian neural network with mini-batches of
size8192andshowthatflipoutspeedsuptrainingintermsofthenumberofiterations.
WetrainedtheFCandConvLenetworksfromSection4.1usingBayesbyBackprop(Blundelletal.,
2015). Since our primary focus is optimization, we focus on the training loss, shown in Fig. 2a:
forFC,wecompareflipoutwithsharedperturbationsandtheLRT;forConvLe, wecompareonly
tosharedperturbationssincetheLRTdoesnotgiveanunbiasedgradientestimator. Wefoundthat
flipoutconvergedinabout3timesfeweriterationsthansharedperturbationsforbothmodels,while
achievingcomparableperformancetotheLRTfortheFCmodel. Becauseflipoutisroughlytwice
as expensive as shared perturbations (see Section 3.1), this corresponds to a 1.5x speedup overall.
CurvesforthetrainingandtesterroraregiveninAppendixE.2.
4.4 EVOLUTIONSTRATEGIES
ES typically runs on multiple CPU cores. The challenge in making ES GPU-friendly is that each
sample requires computing a separate weight perturbation, so traditionally each worker can only
generateonesampleatatime. InSection3.1,weshowedthatESwithflipoutallowseachworkerto
evaluateabatchofperturbations,whichcanbedoneefficientlyonaGPU.However,flipoutinduces
correlationsbetweenthesamples,soweinvestigatedwhetherthesecorrelationscauseaslowdown
in training relative to fully independent perturbations (which we term “IdealES”). In this section,
weshowempiricallythatflipoutESisjustassample-efficientasIdealES,andconsequentlyonecan
obtainsignificantlyhigherthroughputperunitcostusingflipoutESonaGPU.
The ES gradient defined in Eqn. 1 has high variance, so a large number of samples are generally
neededbeforeapplyinganupdate. Wefoundthat5,000samplesareneededtoachievestableper-
formance in the supervised learning tasks. Standard ES runs the forward pass 5,000 times with
independent weight perturbations, which sees little benefit to using a GPU over a CPU. FlipES
allowsthesamenumberofsamplestobeevaluatedusingamuchsmallernumberofexplicitpertur-
bations. Throughouttheexperiments, weranflipoutwithmini-batchesofsize 40(i.e.N = 40in
Eqn.5).
We compared IdealES and FlipES with a fully connected network (FC) on the MNIST dataset.
Fig. 2b shows that we incur no loss in performance when using pseudo-independent noise. Next,
wecomparedFlipESandcpuES(using40CPUcores)intermsoftheper-updatetimewithrespect
tothemodelsize. Theresult(inAppendixE.3)showsthatFlipESscalesbetterbecauseitrunson
theGPU.Finally,wecomparedFlipESandthebackpropagationalgorithmonbothFCandConvLe.
Fig.2candFig.2dshowthatFlipESachievesdataefficiencycomparablewiththebackpropagation
8

PublishedasaconferencepaperatICLR2018
Train Loss (FC) Train Error
| 2.0 | 0.08     |         |
| --- | -------- | ------- |
|     | LRT 0.07 | IdealES |
1.5
|     | NonFlip 0.06 | FlipES |
| --- | ------------ | ------ |
Flip 0 . 0 5
| 1.0 | 0 . 0 4 |     |
| --- | ------- | --- |
0.03
| 0.5 | 0.02 |     |
| --- | ---- | --- |
0 5000 10000 15000 20000 25000 30000 0 2000 4000 6000 8000 10000 12000
Train Loss (Conv) Validation Error
| 20  | 0.08         |         |
| --- | ------------ | ------- |
| 15  | NonFlip 0.07 | IdealES |
|     | Flip 0.06    | FlipES  |
10
0.05
| 5   | 0.04 |     |
| --- | ---- | --- |
| 0   | 0.03 |     |
1000 2000 3000 4000 5000 6000 0 2000 4000 6000 8000 10000 12000
Iterations Iterations
(a)LargeBatchTrainingw/BayesbyBackprop (b)Flipoutvs.FullyIndependentPerturbations
Train Error Train Error
| 0.07    | 0.05              |     |
| ------- | ----------------- | --- |
| 0.06    | FlipES(5000) 0.04 |     |
| 0 . 0 5 | FlipES(1600)      |     |
| 0 . 0 4 | 0.03              |     |
| 0 . 0 3 | 0.02              |     |
0 . 0 2 Backprop
| 0.01 | 0.01 FlipES(5000) |     |
| ---- | ----------------- | --- |
| 0.00 | 0.00              |     |
0 2500 5000 7500 10000 12500 15000 17500 20000 0 500 1000 1500 2000 2500 3000 3500 4000
Validation Error Validation Error
| 0.07 | 0.05     |     |
| ---- | -------- | --- |
| 0.06 | Backprop |     |
0.04
| 0.05 | 0.03 |     |
| ---- | ---- | --- |
0.04
| 0.03 | 0.02 |     |
| ---- | ---- | --- |
0.02 FlipES(1600)
0.01
0.01
0 2500 5000 7500 10000 12500 15000 17500 20000 0 500 1000 1500 2000 2500 3000 3500 4000
Iterations Iterations
(c)Backpropvs.FlipES(FC) (d)Backpropvs.FlipES(ConvLe)
Figure2:
LargebatchtrainingandES.a)TraininglossperiterationusingBayesByBackpropwithbatchsize
8192ontheFCandConvLenetworks.b)ErrorrateoftheFCnetworkonMNISTusingESwith1,600samples
perupdate;thereisnodropinperformancecomparedtoidealES.c)ErrorrateofFConMNIST,comparing
FlipES(witheither5,000or1,600samplesperupdate)withbackpropagation.(Thisfiguredoesnotimplythat
FlipESismoreefficientthanbackprop;FlipESwasaround60timesmoreexpensivethanbackpropperupdate.)
d)Thesameas(c),exceptrunonConvLe.
algorithm. IdealES has a much higher computational cost than backpropagation, due to the large
number of forward passes. FlipES narrows the computational gap between them. Although ES is
moreexpensivethanbackpropagation,itcanbeappliedtomodelswhicharenotfullydifferentiable,
suchasmodelswithadiscreteloss(e.g.,accuracyorBLEUscore)orwithstochasticunits.
5 CONCLUSIONS
Wehaveintroducedflipout,anefficientmethodfordecorrelatingtheweightgradientsbetweendif-
ferentexamplesinamini-batch. Weshowedthatflipoutisguaranteedtoreducethevariancecom-
paredwithsharedperturbations. Empirically,wedemonstratedsignificantvariancereductioninthe
largebatchsettingforavarietyofnetworkarchitectures,aswellassignificantspeedupsintraining
time. Weshowedthatflipoutoutperformsdropout-basedmethodsforregularizingLSTMs. Flipout
also makes it practical to apply GPUs to evolution strategies, resulting in substantially increased
throughputforagivencomputationalcost. Webelieveflipoutwillmakeweightperturbationsprac-
tical in the large batch setting favored by modern accelerators such as Tensor Processing Units
(Jouppietal.,2017).
ACKNOWLEDGMENTS
YWwassupportedbyanNSERCUSRAaward, andPVwassupportedbyaConnaughtNewRe-
searcherAward. WethankDavidDuvenaud,AlexGraves,GeoffreyHinton,andMatthewD.Hoff-
manforhelpfuldiscussions.
9

PublishedasaconferencepaperatICLR2018
REFERENCES
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty
inneuralnetworks. InProceedingsofthe32ndInternationalConferenceonMachineLearning
(ICML),pp.1613–1622,2015.
Tim Cooijmans, NicolasBallas, Ce´sar Laurent, C¸ag˘lar Gu¨lc¸ehre, and AaronCourville. Recurrent
batchnormalization. InInternationalConferenceonLearningRepresentations(ICLR),2017.
MeireFortunato,MohammadGheshlaghiAzar,BilalPiot,JacobMenick,IanOsband,AlexGraves,
VladMnih,RemiMunos,DemisHassabis,OlivierPietquin,etal.Noisynetworksforexploration.
arXivpreprintarXiv:1706.10295,2017.
Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent
neuralnetworks. InAdvancesinNeuralInformationProcessingSystems(NIPS),pp.1019–1027,
2016.
AlexGraves.Practicalvariationalinferenceforneuralnetworks.InAdvancesinNeuralInformation
ProcessingSystems(NIPS),pp.2348–2356,2011.
DavidHa,AndrewDai,andQuocVLe. Hypernetworks. arXivpreprintarXiv:1609.09106,2016.
Geoffrey E Hinton and Drew Van Camp. Keeping the neural networks simple by minimizing the
descriptionlengthoftheweights.InProceedingsofthe6thAnnualConferenceonComputational
LearningTheory,pp.5–13.ACM,1993.
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducinginternalcovariateshift. InInternationalConferenceonMachineLearning(ICML),pp.
448–456,2015.
NormanP.Jouppi,CliffYoung,NishantPatil,DavidPatterson,GauravAgrawal,RaminderBajwa,
Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Pierre luc Cantin, Clifford
Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir
Ghaemmaghami, Rajendra Gottipati, William Gulland, Robert Hagmann, C. Richard Ho, Doug
Hogberg,JohnHu,RobertHundt,DanHurt,JulianIbarz,AaronJaffey,AlekJaworski,Alexander
Kaplan, Harshit Khaitan, Andy Koch, Naveen Kumar, Steve Lacy, James Laudon, James Law,
Diemthu Le, Chris Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin, Gordon MacKean, Adriana
Maggiore,MaireMahony,KieranMiller,RahulNagarajan,RaviNarayanaswami,RayNi,Kathy
Nix, Thomas Norrie, Mark Omernick, Narayana Penukonda, Andy Phelps, and Jonathan Ross.
In-datacenterperformanceanalysisofatensorprocessingunit. 2017. URLhttps://arxiv.
org/pdf/1704.04760.pdf.
DiederikPKingmaandMaxWelling. Auto-encodingvariationalBayes. InProceedingsofthe2nd
InternationalConferenceonLearningRepresentations(ICLR),2014.
DiederikPKingma,TimSalimans,andMaxWelling. Variationaldropoutandthelocalreparame-
terizationtrick. InAdvancesinNeuralInformationProcessingSystems(NIPS),2015.
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. In
TechnicalReport.UniversityofToronto,2009.
David Krueger, Tegan Maharaj, Ja´nos Krama´r, Mohammad Pezeshki, Nicolas Ballas, Nan Rose-
maryKe, AnirudhGoyal, YoshuaBengio, HugoLarochelle, AaronC.Courville, andChrisPal.
Zoneout:RegularizingRNNsbyrandomlypreservinghiddenactivations.CoRR,abs/1606.01305,
2016.
Quoc Le, Tama´s Sarlo´s, and Alex Smola. Fastfood-approximating kernel expansions in loglinear
time. InProceedingsoftheInternationalConferenceonMachineLearning(ICLR),2013.
YannLeCun,Le´onBottou,YoshuaBengio,andPatrickHaffner. Gradient-basedlearningappliedto
documentrecognition. ProceedingsoftheIEEE,86(11):2278–2324,1998.
ChristosLouizos,KarenUllrich,andMaxWelling. Bayesiancompressionfordeeplearning. arXiv
preprintarXiv:1705.08665,2017.
10

PublishedasaconferencepaperatICLR2018
Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated
corpusofEnglish: ThePennTreebank. ComputationalLinguistics,19(2):313–330,1993.
StephenMerity,NitishSKeskar,andRichardSocher. RegularizingandoptimizingLSTMlanguage
models. arXivpreprintarXiv:1708.02182,2017.
AndrewCMiller, NicholasJFoti, AlexanderD’Amour, andRyanPAdams. Reducingreparame-
terizationgradientvariance. arXivpreprintarXiv:1705.07880,2017.
AndriyMnihandKarolGregor. Neuralvariationalinferenceandlearninginbeliefnetworks. arXiv
preprintarXiv:1402.0030,2014.
Matthias Plappert, Rein Houthooft, Prafulla Dhariwal, Szymon Sidor, Richard Y Chen, Xi Chen,
TamimAsfour,PieterAbbeel,andMarcinAndrychowicz. Parameterspacenoiseforexploration.
arXivpreprintarXiv:1706.01905,2017.
Rajesh Ranganath, Sean Gerrish, and David Blei. Black box variational inference. In Artificial
IntelligenceandStatistics(AISTATS),pp.814–822,2014.
IngoRechenbergandManfredEigen. Evolutionsstrategie: OptimierungTechnischerSystemenach
Prinzipien der Biologischen Evolution. Friedrich Frommann Verlag, Stuttgart-Bad Cannstatt,
1973.
Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of Adam and beyond. In
InternationalConferenceonLearningRepresentations(ICLR),2018.
Geoffrey Roeder, Yuhuai Wu, and David Duvenaud. Sticking the landing: A simple, reduced-
variance gradient estimator for variational inference. In Advances in Approximate Bayesian In-
ferenceWorkshop(NIPS),2016.
TimSalimans,JonathanHo,XiChen,andIlyaSutskever. Evolutionstrategiesasascalablealterna-
tivetoreinforcementlearning. arXivpreprintarXiv:1703.03864,2017.
Ju¨rgen Schmidhuber, Daan Wierstra, Matteo Gagliolo, and Faustino Gomez. Training recurrent
networksbyevolino. NeuralComputation,19(3):757–779,2007.
StanislauSemeniuta,AliakseiSeveryn,andErhardtBarth. Recurrentdropoutwithoutmemoryloss.
In Proceedings of the 26th International Conference on Computational Linguistics (COLING),
pp.1757–1766,2016.
KarenSimonyanandAndrewZisserman. Verydeepconvolutionalnetworksforlarge-scaleimage
recognition. arXivpreprintarXiv:1409.1556,2014.
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout:Asimplewaytopreventneuralnetworksfromoverfitting.JournalofMachineLearning
Research,15:1929–1958,2014.
Li Wan, Matthew Zeiler, Sixin Zhang, Yann L Cun, and Rob Fergus. Regularization of neural
networksusingDropConnect. InProceedingsofthe30thInternationalConferenceonMachine
Learning(ICML),pp.1058–1066,2013.
RonaldJWilliams. Simplestatisticalgradient-followingalgorithmsforconnectionistreinforcement
learning. MachineLearning,8(3-4):229–256,1992.
Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization.
arXivpreprintarXiv:1409.2329,2014.
11

PublishedasaconferencepaperatICLR2018
A PROOF OF THEOREM 2
Inthissection,weprovidetheproofofTheorem2(VarianceDecompositionTheorem).
Proof. We use the notations from Section 3.2. Let x,x(cid:48) denote two training examples from the
mini-batch B, and ∆W,∆W(cid:48) denote the weight perturbations they received. We begin with the
decompositionintodataandestimationterms(Eqn.6),whichwerepeathereforconvenience:
(cid:18) (cid:2) (cid:3) (cid:19) (cid:104) (cid:0) (cid:1)(cid:105)
Var(G )=Var E G |B +E Var G |B . (13)
B B B
B ∆W B ∆W
(cid:124) (cid:123)(cid:122) (cid:125) (cid:124) (cid:123)(cid:122) (cid:125)
data estimation
ThedatatermfromEqn.13canbesimplified:
(cid:18) (cid:2) (cid:3) (cid:19) (cid:32) (cid:104) 1 (cid:88) N (cid:12) (cid:105) (cid:33)
V
B
ar
∆
E
W
G
B
|B =V
B
ar
∆
E
W N
G
xn
(cid:12)B
n=1
(cid:32) N (cid:33)
1 (cid:88) (cid:2) (cid:3)
=Var E G |x
B N ∆W xn n
n=1
(cid:18) (cid:19)
1 (cid:2) (cid:3)
= Var E G |x (14)
N x ∆W x
WebreaktheestimationtermfromEqn.13intovarianceandcovarianceterms:
(cid:34) (cid:32) N (cid:33)(cid:35)
(cid:104) (cid:0) (cid:1)(cid:105) 1 (cid:88) (cid:12)
E
B
V
∆
a
W
r G
B
|B =E
B
V
∆
a
W
r
N
G
xn
(cid:12)x
n
n=1
(cid:34) N N (cid:35)
1 (cid:88) (cid:88) (cid:0) (cid:1)
= E Cov G ,G |x ,x
N2 B
n=1n(cid:48)=1
∆Wn,∆W n(cid:48) xn x n(cid:48) n n(cid:48)
 
N
1 (cid:88) (cid:88) (cid:0) (cid:1)
= N2 E B  n=1 ∆ V W ar n (G xn |x n )+ n(cid:54)=n(cid:48) ∆W C n, o ∆ v W n(cid:48) G xn ,G x n(cid:48) |x n ,x n(cid:48) 
1 (cid:104) (cid:105) N −1 (cid:20) (cid:21)
= E Var(G |x) + E Cov (G ,G |x,x(cid:48)) (15)
N x ∆W x N x,x(cid:48) ∆W,∆W(cid:48) x x(cid:48)
Wenowseparatelyanalyzethecasesoffullyindependentperturbations, sharedperturbations, and
flipout.
Fully independent perturbations. If the perturbations are fully independent, the second term in
Eqn.15disappears. Hence,combiningEqns.13,14,and15,weareleftwith
1 (cid:18) (cid:2) (cid:3) (cid:19) 1 (cid:104) (cid:105)
Var(G )= Var E G |x + E Var(G |x) , (16)
B N x ∆W x N x ∆W x
whichisjustα/N.
Shared perturbations. Recall that we reformulate the shared perturbations in terms of first sam-
pling∆(cid:100)W,andthenletting∆W = ∆(cid:100)W ◦rs(cid:62),wherer andsarerandomsignvectorssharedby
thewholebatch. UsingtheLawofTotalVariance,webreakthesecondterminEqn.15intoapart
thatcomesfromsampling∆(cid:100)W andapartthatcomesfromsamplingrands.
(cid:20) (cid:21)
Cov (G
x
,G
x(cid:48)
|x,x(cid:48))= E Cov (G
x
,G
x(cid:48)
|x,x(cid:48),∆(cid:100)W) (cid:12) (cid:12)x,x(cid:48) +
∆W,∆W(cid:48) ∆W,∆W(cid:48)
∆(cid:100)W
(cid:18) (cid:19)
+Cov E [G
x
|x,∆(cid:100)W], E [G
x(cid:48)
|x(cid:48),∆(cid:100)W] (cid:12) (cid:12)x,x(cid:48) (17)
∆(cid:100)W ∆W ∆W(cid:48)
12

PublishedasaconferencepaperatICLR2018
Sincetheperturbationsareshared,∆W(cid:48) =∆W,sothiscanbesimplifiedslightlyto:
|           |     |     |          |           | (cid:18) |     |     |     | (cid:19) |
| --------- | --- | --- | -------- | --------- | -------- | --- | --- | --- | -------- |
| (cid:104) |     |     | (cid:12) | (cid:105) |          |     |     |     | (cid:12) |
E Cov(G ,G |x,x(cid:48),∆(cid:100)W) (cid:12)x,x(cid:48) +Cov E [G |x,∆(cid:100)W], E [G |x(cid:48),∆(cid:100)W] (cid:12)x,x(cid:48) (18)
|                | x   | x(cid:48) |     |     |                | x   |     | x(cid:48) |     |
| -------------- | --- | --------- | --- | --- | -------------- | --- | --- | --------- | --- |
| ∆(cid:100)W ∆W |     |           |     |     | ∆(cid:100)W ∆W |     |     | ∆W        |     |
Plugging these two terms into the second term of Eqn. 15 yields N−1(β +γ), so putting this all
N
| togetherwegetVar(G |     | )=  | 1α+ | N−1(β+γ). |     |     |     |     |     |
| ------------------ | --- | --- | --- | --------- | --- | --- | --- | --- | --- |
B
|     |     |     | N   | N   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Flipout. Since the perturbations for different examples are independent conditioned on ∆(cid:100)W, the
first term of Eqn. 17 vanishes. However, the second term remains. Therefore, plugging this into
Eqn.15andcombiningtheresultwithEqns.13and14,weareleftwithVar(G )= 1α+ N−1γ.
|           |     |                |     |     |     |     |     | B   | N N |
| --------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
| B NETWORK |     | CONFIGURATIONS |     |     |     |     |     |     |     |
Here,weprovidedetailsofthenetworkconfigurationsusedforourexperiments(Section4).
TheFCnetworkisa3-layerfully-connectednetworkwith512-512-10hiddenunits.
ConvLe is a LeNet-like network (LeCun et al., 1998) where the first two layers are convolutional
with32and64filtersofsize[5,5],anduseReLUnon-linearities.A2×2maxpoolinglayerfollows
aftereachconvolutionallayer. Dimensionalityreductiononlytakesplaceinthepoolinglayer; the
strideforpoolingistwoandpaddingisusedintheconvolutionallayerstokeepthedimension. Two
fully-connectedlayerswith1024and10hiddenunitsareusedtoproducetheclassificationresult.
ConVGG is based on the VGG16 network (Simonyan & Zisserman, 2014). We modified the last
fully connected layer to have 10 output dimensions for our experiments on CIFAR-10. We didn’t
usebatchnormalizationforthevariancereductionexperimentsinceitintroducesextrastochasticity.
ThearchitecturesusedfortheLSTMexperimentsaredescribedinSection4.2.Thehyperparameters
usedforthelanguagemodellingexperimentsareprovidedinAppendixD.
| C VARIANCE |     | REDUCTION |     | EXPERIMENT |     | DETAILS |     |     |     |
| ---------- | --- | --------- | --- | ---------- | --- | ------- | --- | --- | --- |
Givenanetworkarchitecture,wecomputetheempiricalstochasticgradientupdatevarianceasfol-
lows. Westartwithamoderatelypre-trainedmodel,suchasanetworkwith85%trainingaccuracy
onMNIST.Withoutupdatingtheparameters,weobtainthegradientsofalltheweightsbyperform-
ingafeed-forwardpass,thatincludessampling∆(cid:100)W,R,andS,followedbybackpropagation. The
gradientvarianceofeachweightiscomputedbyrepeatingthisprocedure200timesintheexperi-
(cid:93)
ments. LetVar lj denotetheestimateofthegradientvarianceofweightjinlayerl. Wecomputethe
gradientvarianceasfollows:
|     |     |          | 1 200    |     |     |       |     | 1 200       |     |
| --- | --- | -------- | -------- | --- | --- | ----- | --- | ----------- | --- |
|     |     | (cid:93) | (cid:88) | (gi | )2  |       |     | (cid:88) gi |     |
|     |     | V ar =   |          | −g  |     | where | g = |             |     |
|     |     | lj       | 200      | lj  | lj  |       | lj  | 200 lj      |     |
|     |     |          | i=1      |     |     |       |     | i=1         |     |
wheregi isthegradientreceivedbyweightj inlayerl. Weestimatethevarianceofthegradients
lj
(cid:80) (cid:93)
in layer l by averaging the variances of the weights in that layer, V˜ = 1 V ar . In order to
|     |     |     |     |     |     |     |     | |J| j | lj  |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
computeaconfidenceintervalonthegradientvarianceestimate,werepeattheaboveprocedure50
times,yieldingasequenceofaveragevarianceestimates,V(cid:102)1 ,...,V(cid:102)50 . ForFig.1,wecomputethe
90%confidenceintervalsofthevarianceestimateswithat-test.
For ConVGG, multiple GPUs were needed to run the variance reduction experiment with large
mini-batchsizes(suchas4096and8192). Insuchcases,itiscomputationallyefficienttogenerate
independent weight perturbations on different GPUs. However, since our aim was to understand
the effects of variance reduction independent of implementation, we shared the base perturbation
among all GPUs to produce the plot shown in Fig. 1. We show in Appendix E that flipout yields
lowervarianceevenwhenwesampleindependentperturbationsondifferentGPUs.
For the LSTM variance reduction experiments, we used the two-layer LSTM described in Sec-
tion 4.2, trained for 3 epochs on the word-level Penn Treebank dataset. For Fig. 1, we split large
13

PublishedasaconferencepaperatICLR2018
mini-batches (size 128 and higher) into sub-batches of size 64; we sampled one base perturbation
∆W that was shared among all sub-batches, and we sampled independent R and S matrices for
eachsub-batch.
| D   | LSTM | REGULARIZATION |     |     | EXPERIMENT |     | DETAILS |     |     |
| --- | ---- | -------------- | --- | --- | ---------- | --- | ------- | --- | --- |
LongShort-TermMemorynetworks(LSTMs)aredefinedbythefollowingequations:
|     |     |     | i ,f | ,o =σ(W   |           | h     | +W  | x +b)  | (19) |
| --- | --- | --- | ---- | --------- | --------- | ----- | --- | ------ | ---- |
|     |     |     | t    | t t       |           | h t−1 | x   | t      |      |
|     |     |     |      | g =tanh(W |           | h     | +U  | x +b ) | (20) |
|     |     |     |      | t         |           | g     | t−1 | g t g  |      |
|     |     |     |      | c =f      | ◦c        | +i    | ◦g  |        | (21) |
|     |     |     |      | t         | t         | t−1   | t t |        |      |
|     |     |     |      | h t =o    | t ◦tanh(c |       | t ) |        | (22) |
wherei t ,f t ,ando t aretheinput,forget,andoutputgates,respectively,g t isthecandidateupdate,
and ◦ denotes elementwise multiplication. Na¨ıve application of dropout on the hidden state of an
LSTM is not effective, because it leads to significant memory loss over long sequences. Several
approacheshavebeenproposedtoregularizetherecurrentconnections,basedonapplyingdropout
to specific terms in the LSTM equations. Semeniuta et al. (2016) propose to drop the cell update
vector,withadropoutmaskd t sampledeitherper-steporper-sequence:c t =f t ◦c t−1 +i t ◦(d t ◦g t ).
Gal&Ghahramani(2016)applydropouttotheinputandhiddenstateateachtimestep,x ◦d and
t x
h ◦d ,withdropoutmasksd andd sampledoncepersequence(andrepeatedineachtime
| t−1 | h   |     |     | x   | h   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
step). Krueger et al. (2016) propose to zone out units rather than dropping them; the hidden state
andcellvaluesareeitherstochasticallyupdatedormaintaintheirpreviousvalue: c =dc◦c +
t t t−1
| (1−dc)◦(f |     |     |     |     | =dh◦h |     | +(1−dh)◦(o |     |     |
| --------- | --- | --- | --- | --- | ----- | --- | ---------- | --- | --- |
t ◦c t−1 +i t ◦g t )andh t t−1 t ◦tanh(f t ◦c t−1 +i t ◦g t )),
|     | t   |     |     |     | t   |     |     | t   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
withzoneoutmasksdhanddcsampledperstep.
|     |     | t   | t   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
D.1 HYPERPARAMETERDETAILS
For the word-level models (Table 3), we used gradient clipping threshold 0.25 and the following
hyperparameters:
• For Gal & Ghahramani (2016), we used variational dropout with the parameters given in
theirpaper: 0.35dropoutprobabilityoninputsandoutputs,0.2hiddenstatedropout,and
0.2embeddingdropout.
• For Semeniuta et al. (2016), we used 0.1 embedding dropout, 0.5 dropout on inputs and
outputs,and0.3dropoutoncellupdates,withper-stepmasksampling.
• For Krueger et al. (2016), we used 0.1 embedding dropout, 0.5 dropout on inputs and
outputs,andcellandhiddenstatezoneoutprobabilitiesof0.25and0.025,respectively.
• ForWD(Merityetal.,2017),weusedtheparametersgivenintheirpaper: 0.1embedding
dropout,0.4dropoutprobabilityoninputsandoutputs,and0.3dropoutprobabilityonthe
outputbetweenlayers(thesamemasksareusedforeachstepofasequence). Weuse0.5
probabilityforDropConnectappliedtothehidden-to-hiddenweightmatrices.
• For WD+Flipout, we used the same parameters as Merity et al. (2017), given above, but
weregularizedthehidden-to-hiddenweightmatriceswiththevariantofflipoutdescribed
inSection2.1,whichimplementsDropConnectwithprobability0.5.
For the character-level models (Table 2), we used orthogonal initialization for the LSTM weight
matrices,gradientclippingthreshold1,anddidnotuseinputoroutputdropout.Theinputcharacters
wererepresentedasone-hotvectors. Weusedthefollowinghyperparametersforeachmodel:
• Forrecurrentdropout(Semeniutaetal.,2016),weused0.25dropoutprobabilityonthecell
state,andper-stepmasksampling.
• For Zoneout (Krueger et al., 2016), we used 0.5 and 0.05 for the cell and hidden state
zoneoutprobabilities,respectively.
• ForthevariationalLSTM(Gal&Ghahramani,2016),weused0.25hiddenstatedropout.
• FortheflipoutandsharedperturbationLSTMs,wesampledGaussiannoisewithσ =1for
thehidden-to-hiddenweightmatrix.
14

PublishedasaconferencepaperatICLR2018
| E ADDITIONAL | EXPERIMENTS |     |     |     |     |     |
| ------------ | ----------- | --- | --- | --- | --- | --- |
E.1 VARIANCEREDUCTION
As discussed in Appendix B, training on multiple GPUs naturally induces independent noise for
each sub-batch. Fig. 3 shows that flipout still achieves lower variance than shared perturbations
in such cases. When estimating the variance with mini-batch size 8192, running on four GPUs
naturallyinducesfourindependentnoisesamples,foreachsub-batchofsize2048;thisyieldslower
variance than using a single noise sample. Similarly, for mini-batch size 4096, two independent
noisesamplesaregeneratedonseparateGPUs.
Variance Estimation
102
103
ecnairav
104
105
106
Conv1
|     | 107 | Conv8   |     |     |     |     |
| --- | --- | ------- | --- | --- | --- | --- |
|     | 100 | 101 102 | 103 | 104 |     |     |
batch size
Figure3: EmpiricalvarianceofthegradientswhentrainingonmultipleGPUs.Solid:flipout.Dotted:shared
perturbations.
E.2 LARGEBATCHTRAININGWITHFLIPOUT
Fig.4showsthetrainingandtesterrorforthelargemini-batchexperimentsdescribedinSection4.3.
For both FC and ConvLe networks, we used the Adam optimizer with learning rate 0.003. We
downscaledtheKLtermbyafactorof10toachievehigheraccuracy.
WhileFig.2ashowsthatflipoutconvergesfasterthansharedperturbations,Fig.4showsthatflipout
hasthesamegeneralizationabilityassharedperturbations(thefasterconvergencedoesn’tresultin
overfitting).
|        | Train Error (FC) |            |        | Train Error (Conv) |           |         |
| ------ | ---------------- | ---------- | ------ | ------------------ | --------- | ------- |
| 0.04   |                  |            | 0.04   |                    |           |         |
|        |                  | NonFlip    |        |                    |           | NonFlip |
| 0.03   |                  |            | 0.03   |                    |           |         |
|        |                  | Flip       |        |                    |           | Flip    |
| 0.02   |                  |            | 0.02   |                    |           |         |
| 0.01   |                  |            | 0.01   |                    |           |         |
| 0.00   |                  |            | 0.00   |                    |           |         |
| 0 2000 | 4000 6000        | 8000 10000 | 0 1000 | 2000 3000          | 4000 5000 | 6000    |
|        | Test Error (FC)  |            |        | Test Error (Conv)  |           |         |
| 0.04   |                  |            | 0.04   |                    |           |         |
|        |                  | NonFlip    |        |                    |           | NonFlip |
0.03
| 0.03 |     | Flip |     |     |     | Flip |
| ---- | --- | ---- | --- | --- | --- | ---- |
0.02
0.02
0.01
| 0.01   |            |            | 0.00   |            |           |      |
| ------ | ---------- | ---------- | ------ | ---------- | --------- | ---- |
| 0 2000 | 4000 6000  | 8000 10000 | 0 1000 | 2000 3000  | 4000 5000 | 6000 |
|        | Iterations |            |        | Iterations |           |      |
Figure4: Left: ThetrainingandtesterrorsobtainedbytrainingtheFCnetworkonlargemini-batches(size
8192)withBayesbyBackprop. Right:ThetrainingandtesterrorsobtainedwithConvLeinthesamesetting,
withmini-batchsize8192.
| E.3 FLIPESV.S. | CPUES |     |     |     |     |     |
| -------------- | ----- | --- | --- | --- | --- | --- |
Fig. 5 shows that the computational cost of cpuES increases as the model size increases, while
FlipESscalesbetterbecauseitrunsontheGPU.
15

PublishedasaconferencepaperatICLR2018
40
20
0
0 250 500 750 1000 1250 1500 1750 2000
hidden units
sces
Update time (FC)
FlipES cpuES
200
100
0
0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
model scale
sces
#HiddenUnits FlipES cpuES
32 0.12s 0.51s
128 0.13s 1.22s
512 0.18s 5.10s
2048 1.86s 38.0s
#Filters FlipES cpuES
Update time (Conv)
0.25 2.3s 16s
FlipES 0.75 5.48s 46s cpuES
1.0 7.12s 67s
1.5 11.77s 132s
Figure5:Per-updatetimecomparisonbetweenFlipESand40-corecpuES(5,000samples)w.r.t. the
model size. We scale the FC network by modifying the number of hidden units, and we scale the
Convnetworkbymodifyingthenumberoffilters(1.0standsfor32filtersinthefirstconvolutional
layerand64filtersforthesecondone).
E.4 LARGEBATCHLSTMTRAINING
The variance reduction offered by flipout allows us to use DropConnect (Wan et al., 2013) effi-
cientlyinalargemini-batchsetting. Here,weuseflipouttoimplementDropConnectasdescribed
in Section 2.1, and use it to regularize an LSTM word-level language model. We used the LSTM
architecture proposed by Merity et al. (2017), which has 400-dimensional word embedddings and
three layers with hidden dimension 1150. Following Merity et al. (2017), we tied the weights of
theembeddinglayerandthedecoderlayer. Merityetal.(2017)useDropConnecttoregularizethe
hidden-to-hiddenweightmatrices,withasinglemasksharedforallexamplesinabatch. Weused
flipout to achieve a different DropConnect mask per example. We applied WD+Flipout to both
thehidden-to-hidden(h2h)andinput-to-hidden(i2h)weightmatrices, andcomparedtothemodel
fromMerityetal.(2017), whichwecallWD(forweight-dropped LSTM),withDropConnectap-
plied to both h2h and i2h. Both models use embedding dropout 0.1, output dropout 0.4, and have
DropConnect probability 0.5 for the i2h and h2h weights. Both models were trained using Adam
withlearningrate0.001.
Fig. 6 compares the variance of the gradients of the first-layer hidden-to-hidden weights between
WDandWD+Flipout,andshowsthatflipoutachievessignificantvariancereductionformini-batch
sizeslargerthan256. Fig.7showsthetrainingcurvesofbothmodelswithbatchsize8192. Wesee
that WD+Flipout converges faster than WD, and achieves a lower training perplexity, showcasing
theoptimizationbenefitsofflipoutinlargemini-batchsettings.
10 8
10 9
10 10
10 11
10 12
102 103 104
Batch Size
ecnairaV
300
250
200
150
Wf
Wi 100
Wo
Wc 50
0 200 400 600 800 1000
Epoch
Figure6: Thevariancereductionofferedbyflipout
compared to the WD model (Merity et al., 2017).
Solid lines represent WD+Flipout, while dotted
lines represent WD. The variance is shown for the
hidden-to-hiddenweightmatricesinthefirstlayer:
W , W , W , and W are the weights for the for-
f i o c
get, input and output gates, and the candidate cell
update,respectively.
ytixelpreP
niarT
WD
WD+Flipout
Figure 7: Training curves for WD and
WD+Flipout,withbatchsize8192.
16