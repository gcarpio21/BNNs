|                  |     |     |     | Weight | Uncertainty |     | in Neural | Networks |     |                      |                     |     |     |
| ---------------- | --- | --- | --- | ------ | ----------- | --- | --------- | -------- | --- | -------------------- | ------------------- | --- | --- |
| CharlesBlundell  |     |     |     |        |             |     |           |          |     | CBLUNDELL@GOOGLE.COM |                     |     |     |
| JulienCornebise  |     |     |     |        |             |     |           |          |     |                      | JUCOR@GOOGLE.COM    |     |     |
| KorayKavukcuoglu |     |     |     |        |             |     |           |          |     |                      | KORAYK@GOOGLE.COM   |     |     |
| DaanWierstra     |     |     |     |        |             |     |           |          |     |                      | WIERSTRA@GOOGLE.COM |     |     |
GoogleDeepMind
5102 yaM 12  ]LM.tats[  2v42450.5051:viXra
|     |           |        | Abstract   |            |     |     | ventoverfittinginneuralnetworkssuchasearlystopping, |           |             |               |            |             |         |
| --- | --------- | ------ | ---------- | ---------- | --- | --- | --------------------------------------------------- | --------- | ----------- | ------------- | ---------- | ----------- | ------- |
|     |           |        |            |            |     |     | weight decay,                                       |           | and dropout |               | (Hinton et | al., 2012). | In this |
| We  | introduce | a new, | efficient, | principled |     | and |                                                     |           |             |               |            |             |         |
|     |           |        |            |            |     |     | work, we                                            | introduce |             | an efficient, | principled | algorithm   | for     |
backpropagation-compatiblealgorithmforlearn-
regularisationbuiltuponBayesianinferenceontheweights
| ing         | a probability |             | distribution | on the          | weights   | of   |                                |     |           |         |                    |          |          |
| ----------- | ------------- | ----------- | ------------ | --------------- | --------- | ---- | ------------------------------ | --- | --------- | ------- | ------------------ | -------- | -------- |
|             |               |             |              |                 |           |      | of the network                 |     | (MacKay,  |         | 1992; Buntine      | and      | Weigend, |
| a neural    | network,      |             | called       | Bayes by        | Backprop. | It   |                                |     |           |         |                    |          |          |
|             |               |             |              |                 |           |      | 1991; MacKay,                  |     | 1995).    | This    | leads to           | a simple | approxi- |
| regularises |               | the weights |              | by minimising   | a         | com- |                                |     |           |         |                    |          |          |
|             |               |             |              |                 |           |      | mate learning                  |     | algorithm | similar | to backpropagation |          | (Le-     |
| pression    | cost,         | known       | as           | the variational | free      | en-  |                                |     |           |         |                    |          |          |
|             |               |             |              |                 |           |      | Cun,1985;Rumelhartetal.,1988). |     |           |         | Weshalldemonstrate |          |          |
ergyortheexpectedlowerboundonthemarginal
|             |     |         |      |                 |     |      | how this | uncertainty |     | can improve | predictive |     | performance |
| ----------- | --- | ------- | ---- | --------------- | --- | ---- | -------- | ----------- | --- | ----------- | ---------- | --- | ----------- |
| likelihood. |     | We show | that | this principled |     | kind |          |             |     |             |            |     |             |
inregressionproblemsbyexpressinguncertaintyinregions
ofregularisationyieldscomparableperformance
withlittleornodata,howthisuncertaintycanleadtomore
| to          | dropout | on MNIST |            | classification. | We  | then   |            |             |     |      |                    |            |        |
| ----------- | ------- | -------- | ---------- | --------------- | --- | ------ | ---------- | ----------- | --- | ---- | ------------------ | ---------- | ------ |
|             |         |          |            |                 |     |        | systematic | exploration |     | than | (cid:15)-greedy in | contextual | bandit |
| demonstrate |         | how      | the learnt | uncertainty     |     | in the |            |             |     |      |                    |            |        |
tasks.
| weights | can | be used | to  | improve generalisation |     |     |     |     |     |     |     |     |     |
| ------- | --- | ------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
in non-linear regression problems, and how this Allweightsinourneuralnetworksarerepresentedbyprob-
weight uncertainty can be used to drive the abilitydistributionsoverpossiblevalues,ratherthanhaving
exploration-exploitation trade-off in reinforce- a single fixed value as is the norm (see Figure 1). Learnt
mentlearning. representationsandcomputationsmustthereforeberobust
|     |     |     |     |     |     |     | under perturbation |      | of     | the weights, | but     | the amount | of per-       |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | ---- | ------ | ------------ | ------- | ---------- | ------------- |
|     |     |     |     |     |     |     | turbation          | each | weight | exhibits     | is also | learnt     | in a way that |
1.Introduction coherently explains variability in the training data. Thus
|     |     |     |     |     |     |     | instead | of training | a   | single | network, the | proposed | method |
| --- | --- | --- | --- | --- | --- | --- | ------- | ----------- | --- | ------ | ------------ | -------- | ------ |
Plain feedforward neural networks are prone to overfit- trainsanensembleofnetworks,whereeachnetworkhasits
ting. When applied to supervised or reinforcement learn- weights drawn from a shared, learnt probability distribu-
| ing problems | these | networks |     | are also often | incapable |     | of  |     |     |     |     |     |     |
| ------------ | ----- | -------- | --- | -------------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
tion. Unlikeotherensemblemethods,ourmethodtypically
correctlyassessingtheuncertaintyinthetrainingdataand only doubles the number of parameters yet trains an infi-
somakeoverlyconfidentdecisionsaboutthecorrectclass, niteensembleusingunbiasedMonteCarloestimatesofthe
| prediction | or action. | We  | shall | address | both of | these con- |     |     |     |     |     |     |     |
| ---------- | ---------- | --- | ----- | ------- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
gradients.
| cerns by                             | using | variational | Bayesian | learning                 | to           | introduce |                |       |                |            |               |          |              |
| ------------------------------------ | ----- | ----------- | -------- | ------------------------ | ------------ | --------- | -------------- | ----- | -------------- | ---------- | ------------- | -------- | ------------ |
|                                      |       |             |          |                          |              |           | In general,    | exact | Bayesian       |            | inference on  | the      | weights of a |
| uncertaintyintheweightsofthenetwork. |       |             |          |                          | Wecalloural- |           |                |       |                |            |               |          |              |
|                                      |       |             |          |                          |              |           | neural network |       | is intractable |            | as the number | of       | parameters   |
| gorithmBayesbyBackprop.              |       |             |          | Wesuggestatleastthreemo- |              |           |                |       |                |            |               |          |              |
|                                      |       |             |          |                          |              |           | is very large  | and   | the            | functional | form of       | a neural | network      |
tivationsforintroducinguncertaintyontheweights:1)reg-
ularisationviaacompressioncostontheweights,2)richer doesnotlenditselftoexactintegration. Insteadwetakea
|                                   |                |     |           |                   |     |          | variational | approximation |           | to        | exact Bayesian |         | updates. We   |
| --------------------------------- | -------------- | --- | --------- | ----------------- | --- | -------- | ----------- | ------------- | --------- | --------- | -------------- | ------- | ------------- |
| representationsandpredictionsfrom |                |     |           | cheapmodelaverag- |     |          |             |               |           |           |                |         |               |
|                                   |                |     |           |                   |     |          | build upon  | the           | work      | of Graves | (2011),        | who     | in turn built |
| ing, and                          | 3) exploration |     | in simple | reinforcement     |     | learning |             |               |           |           |                |         |               |
|                                   |                |     |           |                   |     |          | upon the    | work          | of Hinton | and       | Van Camp       | (1993). | In con-       |
problemssuchascontextualbandits.
|     |     |     |     |     |     |     | trast to | this previous |     | work, | we show | how | the gradients |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------- | --- | ----- | ------- | --- | ------------- |
Variousregularisationschemeshavebeendevelopedtopre-
|     |     |     |     |     |     |     | of Graves   | (2011) | can     | be made | unbiased     | and | further how  |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | ------- | ------- | ------------ | --- | ------------ |
|     |     |     |     |     |     |     | this method | can    | be used | with    | non-Gaussian |     | priors. Con- |
32nd
Proceedings of the International Conference on Machine sequently,BayesbyBackpropattainsperformancecompa-
| Learning,Lille,France,2015. |     |     | JMLR:W&CPvolume37. |     |     | Copy- |                                         |     |     |     |     |     |           |
| --------------------------- | --- | --- | ------------------ | --- | --- | ----- | --------------------------------------- | --- | --- | --- | --- | --- | --------- |
|                             |     |     |                    |     |     |       | rabletothatofdropout(Hintonetal.,2012). |     |     |     |     |     | Ourmethod |
right2015bytheauthor(s).

WeightUncertaintyinNeuralNetworks
|     |     |     |     |     |     |     | the parameters |                 | of the categorical |          | distribution |                | are | passed |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --------------- | ------------------ | -------- | ------------ | -------------- | --- | ------ |
|     | Y   |     |     |     | Y   |     |                |                 |                    |          |              |                |     |        |
|     |     |     |     |     |     |     | through        | the exponential |                    | function | then         | re-normalised. |     | For    |
isRandP(y|x,w)isaGaussiandistribution
| 0.5 | 0.1 0.7 | 1.3 |     |     |     |     | regressionY |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
–thiscorrespondstoasquaredloss.
| H1  | H2  | H3  | 1   | H1  | H2  | H3  | 1   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.1 0.2 0.1 0.3 1.4 Inputs x are mapped onto the parameters of a distribu-
|     |     |     |     |     |     |     | tiononY | byseveralsuccessivelayersoflineartransforma- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------------------------------------------- | --- | --- | --- | --- | --- | --- |
1.2
tion(givenbyw)interleavedwithelement-wisenon-linear
|     | X   | 1   |     |     | X   | 1   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
transforms.
Theweightscanbelearntbymaximumlikelihoodestima-
Figure1.Left:eachweighthasafixedvalue,asprovidedbyclas-
sicalbackpropagation. Right: eachweightisassignedadistribu- tion(MLE):givenasetoftrainingexamplesD =(x i ,y i ) i ,
tion,asprovidedbyBayesbyBackprop. theMLEweightswMLEaregivenby:
wMLE =argmaxlogP(D|w)
w
| isrelatedtorecentmethodsindeep, |     |     |     | generativemodelling |     |     |     |     |     |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:88)
|                                                 |                                         |       |       |              |     |        |                   |     | =argmax  |             | logP(y | |x      | ,w).   |       |
| ----------------------------------------------- | --------------------------------------- | ----- | ----- | ------------ | --- | ------ | ----------------- | --- | -------- | ----------- | ------ | ------- | ------ | ----- |
| (KingmaandWelling,2014;Rezendeetal.,2014;Gregor |                                         |       |       |              |     |        |                   |     |          | w           |        | i i     |        |       |
| etal.,2014),                                    | wherevariationalinferencehasbeenapplied |       |       |              |     |        |                   |     |          | i           |        |         |        |       |
| to stochastic                                   | hidden                                  | units | of an | autoencoder. |     | Whilst | the               |     |          |             |        |         |        |       |
|                                                 |                                         |       |       |              |     |        | This is typically |     | achieved | by gradient |        | descent | (e.g., | back- |
numberofstochastichiddenunitsmightbeintheorderof
propagation),whereweassumethatlogP(D|w)isdiffer-
| thousands, | the number |     | of weights | in  | a neural | network | is  |     |     |     |     |     |     |     |
| ---------- | ---------- | --- | ---------- | --- | -------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
entiableinw.
easilytwoordersofmagnitudelarger,makingtheoptimisa-
tionproblemmuchlargerscale. Uncertaintyinthehidden Regularisation can be introduced by placing a prior upon
unitsallowstheexpressionofuncertaintyaboutaparticular
|     |     |     |     |     |     |     | the weights | w   | and finding |     | the maximum |     | a posteriori |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ----------- | --- | ----------- | --- | ------------ | --- |
observation, uncertainty in the weights is complementary (MAP)weightswMAP:
inthatitcapturesuncertaintyaboutwhichneuralnetwork
isappropriate,leadingtoregularisationoftheweightsand
wMAP =argmaxlogP(w|D)
| modelaveraging. |     |     |     |     |     |     |     |     | w   |     |     |     |     |     |
| --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
=argmaxlogP(D|w)+logP(w).
Thisuncertaintycanbeusedtodriveexplorationincontex-
w
| tual bandit | problems | using | Thompson |     | sampling | (Thomp- |     |     |     |     |     |     |     |     |
| ----------- | -------- | ----- | -------- | --- | -------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
son, 1933; Chapelle and Li, 2011; Agrawal and Goyal, If w are given a Gaussian prior, this yields L2 regularisa-
2012;Mayetal.,2012). Weightswithgreateruncertainty tion(orweightdecay). IfwaregivenaLaplaceprior,then
introduce more variability into the decisions made by the L1regularisationisrecovered.
network,leadingnaturallytoexploration.Asmoredataare
observed, the uncertainty can decrease, allowing the deci- 3.BeingBayesianbyBackpropagation
| sions made | by the | network | to  | become | more | deterministic |     |     |     |     |     |     |     |     |
| ---------- | ------ | ------- | --- | ------ | ---- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
astheenvironmentisbetterunderstood. Bayesianinferenceforneuralnetworkscalculatesthepos-
|                   |                 |             |               |                |          |           | terior distribution |          | of the       | weights       | given         | the        | training       | data,    |
| ----------------- | --------------- | ----------- | ------------- | -------------- | -------- | --------- | ------------------- | -------- | ------------ | ------------- | ------------- | ---------- | -------------- | -------- |
| The remainder     | of              | the paper   | is            | organised      | as       | follows:  | Sec-                |          |              |               |               |            |                |          |
|                   |                 |             |               |                |          |           | P(w|D).             | This     | distribution |               | answers       | predictive |                | queries  |
| tion 2 introduces |                 | notation    | and           | standard       | learning | in        | neural              |          |              |               |               |            |                |          |
|                   |                 |             |               |                |          |           | about unseen        | data     | by taking    | expectations: |               |            | the predictive |          |
| networks,         | Section         | 3 describes |               | variational    | Bayesian |           | learn-              |          |              |               |               |            |                |          |
|                   |                 |             |               |                |          |           | distribution        | of       | an unknown   | label         | yˆ            | of a test  | data           | item xˆ, |
| ing for neural    | networks        |             | and our       | contributions, |          | Section   | 4                   |          |              |               |               |            |                |          |
|                   |                 |             |               |                |          |           |                     | P(yˆ|xˆ) |              | E             | [P(yˆ|xˆ,w)]. |            |                |          |
|                   |                 |             |               |                |          |           | is given            | by       | =            | P(w|D)        |               |            | Each           | pos-     |
| describes         | the application |             | to contextual |                | bandit   | problems, |                     |          |              |               |               |            |                |          |
|                   |                 |             |               |                |          |           | sible configuration |          | of the       | weights,      | weighted      |            | according      | to       |
whilstSection5containsempiricalresultsonaclassifica-
theposteriordistribution,makesapredictionabouttheun-
| tion,aregressionandabanditproblem. |     |     |     |     | Weconcludewith |     |             |       |     |           |      |          |        |     |
| ---------------------------------- | --- | --- | --- | --- | -------------- | --- | ----------- | ----- | --- | --------- | ---- | -------- | ------ | --- |
|                                    |     |     |     |     |                |     | known label | given | the | test data | item | xˆ. Thus | taking | an  |
abriefdiscussioninSection6.
|     |     |     |     |     |     |     | expectation | under    | the posterior |          | distribution |                | on weights | is    |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | ------------- | -------- | ------------ | -------------- | ---------- | ----- |
|     |     |     |     |     |     |     | equivalent  | to using | an            | ensemble | of           | an uncountably |            | infi- |
2.PointEstimatesofNeuralNetworks
|     |     |     |     |     |     |     | nite number | of  | neural networks. |     | Unfortunately, |     | this | is in- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ---------------- | --- | -------------- | --- | ---- | ------ |
tractableforneuralnetworksofanypracticalsize.
| We view | a neural | network |     | as a | probabilistic |     | model |     |     |     |     |     |     |     |
| ------- | -------- | ------- | --- | ---- | ------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
Rp
P(y|x,w): given an input x ∈ a neural network as- Previously Hinton and Van Camp (1993) and Graves
signs a probability to each possible output y ∈ Y, using (2011)suggestedfindingavariationalapproximationtothe
thesetofparametersorweightsw. Forclassification,Y is Bayesianposteriordistributionontheweights. Variational
asetofclassesandP(y|x,w)isacategoricaldistribution– learning finds the parameters θ of a distribution on the
thiscorrespondstothecross-entropyorsoftmaxloss,when weightsq(w|θ)thatminimisestheKullback-Leibler(KL)

WeightUncertaintyinNeuralNetworks
divergencewiththetrueBayesianposteriorontheweights: The deterministic function t(θ,(cid:15)) transforms a sample of
parameter-freenoise(cid:15)andthevariationalposteriorparam-
θ(cid:63)
=argminKL[q(w|θ)||P(w|D)] etersθintoasamplefromthevariationalposterior. Below
θ
|     |          |     |        |     |     |     | we shall | see how | this transform | works | in  | practice | for the |
| --- | -------- | --- | ------ | --- | --- | --- | -------- | ------- | -------------- | ----- | --- | -------- | ------- |
|     | (cid:90) |     | q(w|θ) |     |     |     |          |         |                |       |     |          |         |
Gaussiancase.
| =argmin                   | q(w|θ)log |     |            |        | dw           |     |          |             |                          |                  |     |         |     |
| ------------------------- | --------- | --- | ---------- | ------ | ------------ | --- | -------- | ----------- | ------------------------ | ---------------- | --- | ------- | --- |
|                           | θ         |     | P(w)P(D|w) |        |              |     |          |             |                          |                  |     |         |     |
|                           |           |     |            |        |              |     | We apply | Proposition | 1 to                     | the optimisation |     | problem | in  |
| =argminKL[q(w|θ)||P(w)]−E |           |     |            |        | [logP(D|w)]. |     |          |             |                          |                  |     |         |     |
|                           |           |     |            | q(w|θ) |              |     | (1): let | f(w,θ) =    | logq(w|θ)−logP(w)P(D|w). |                  |     |         | Us- |
θ
|     |     |     |     |     |     |     | ing Monte | Carlo | sampling | to evaluate | the | expectations, |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | ----- | -------- | ----------- | --- | ------------- | --- |
Theresultingcostfunctionisvariouslyknownasthevaria-
|             |        |           |         |       |         |     | a backpropagation-like |     | (LeCun, | 1985; | Rumelhart |     | et al., |
| ----------- | ------ | --------- | ------- | ----- | ------- | --- | ---------------------- | --- | ------- | ----- | --------- | --- | ------- |
| tional free | energy | (Neal and | Hinton, | 1998; | Yedidia | et  | al.,                   |     |         |       |           |     |         |
1988)algorithmisobtainedforvariationalBayesianinfer-
| 2000; Friston | et al., | 2007) | or the | expected | lower | bound |     |     |     |     |     |     |     |
| ------------- | ------- | ----- | ------ | -------- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
enceinneuralnetworks–BayesbyBackprop–whichuses
(Saul et al., 1996; Neal and Hinton, 1998; Jaakkola and unbiasedestimatesofgradientsofthecostin(1)tolearna
| Jordan,2000). | Forsimplicityweshalldenoteitas |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
distributionovertheweightsofaneuralnetwork.
|     |     |     |     |     |     |     | Proposition | 1 is | a generalisation |     | of the | Gaussian | re- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ---------------- | --- | ------ | -------- | --- |
F(D,θ)=KL[q(w|θ)||P(w)]
|     |     |     |        |              |     |     | parameterisation                                |     | trick (Opper | and | Archambeau, |     | 2009; |
| --- | --- | --- | ------ | ------------ | --- | --- | ----------------------------------------------- | --- | ------------ | --- | ----------- | --- | ----- |
|     |     |     | −E     | [logP(D|w)]. |     |     | (1)                                             |     |              |     |             |     |       |
|     |     |     | q(w|θ) |              |     |     | KingmaandWelling,2014;Rezendeetal.,2014)usedfor |     |              |     |             |     |       |
latentvariablemodels,appliedtoBayesianlearningofneu-
Thecostfunctionof(1)isasumofadata-dependentpart,
|     |     |     |     |     |     |     | ralnetworks. | Ourworkdiffersfromthispreviousworkin |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------------------------------ | --- | --- | --- | --- | --- |
whichweshallrefertoasthelikelihoodcost, andaprior- several significant ways. Bayes by Backprop operates on
| dependent | part, which | we  | shall refer | to  | as the complexity |     |     |     |     |     |     |     |     |
| --------- | ----------- | --- | ----------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
weights(ofwhichthereareagreatmany),whilstmostpre-
cost. Thecostfunctionembodiesatrade-offbetweensatis-
viousworkappliesthismethodtolearningdistributionson
fyingthecomplexityofthedataD andsatisfyingthesim- stochastic hidden units (of which there are far fewer than
| plicitypriorP(w). |     | (1)isalsoreadilygivenaninformation |     |     |     |     |     |     |     |     |     |     |     |
| ----------------- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thenumberofweights).TitsiasandLa´zaro-Gredilla(2014)
theoretic interpretation as a minimum description length considered a large-scale logistic regression task. Unlike
| cost(HintonandVanCamp,1993;Graves,2011). |     |     |     |     |     | Exactly |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
previouswork,wedonotusetheclosedformofthecom-
minimisingthiscostna¨ıvelyiscomputationallyprohibitive.
|     |     |     |     |     |     |     | plexitycost(orentropicpart): |     |     | notrequiringaclosedform |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | ----------------------- | --- | --- | --- |
Instead gradient descent and various approximations are ofthecomplexitycostallowsmanymorecombinationsof
used.
|     |     |     |     |     |     |     | priorandvariationalposteriorfamilies. |     |     |     | Indeedthisscheme |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | ---------------- | --- | --- |
isalsosimpletoimplementandallowsprior/posteriorcom-
3.1.UnbiasedMonteCarlogradients
|     |     |     |     |     |     |     | binations | to be interchanged. |     | We  | approximate | the | exact |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------------------- | --- | --- | ----------- | --- | ----- |
cost(1)as:
| Under certain | conditions, |     | the derivative |     | of an expectation |     |     |     |     |     |     |     |     |
| ------------- | ----------- | --- | -------------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
canbeexpressedastheexpectationofaderivative:
n
(cid:88) logq(w(i)|θ)−logP(w(i))
F(D,θ)≈
| Proposition1.   | Let(cid:15)bearandomvariablehavingaprob- |                |     |       |                 |       |     |     |     |     |     |     |     |
| --------------- | ---------------------------------------- | -------------- | --- | ----- | --------------- | ----- | --- | --- | --- | --- | --- | --- | --- |
| ability density | given                                    | by q((cid:15)) | and | let w | = t(θ,(cid:15)) | where |     | i=1 |     |     |     |     |     |
t(θ,(cid:15)) is a deterministic function. Suppose further that −logP(D|w(i)) (2)
themarginalprobabilitydensityofw,q(w|θ),issuchthat
wherew(i)denotestheithMonteCarlosampledrawnfrom
| q((cid:15))d(cid:15) = | q(w|θ)dw. | Then | for a | function | f with | deriva- |     |     |     |     |     |     |     |
| ---------------------- | --------- | ---- | ----- | -------- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- |
thevariationalposteriorq(w(i)|θ).
Notethateverytermof
tivesinw:
thisapproximatecostdependsupontheparticularweights
(cid:20) (cid:21) drawnfromthevariationalposterior: thisisaninstanceof
| ∂   |            |     | ∂f(w,θ)∂w |     | ∂f(w,θ) |     |     |     |     |     |     |     |     |
| --- | ---------- | --- | --------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
| E   | [f(w,θ)]=E |     |           |     | +       |     | .   |     |     |     |     |     |     |
∂θ q(w|θ) q((cid:15)) ∂w ∂θ ∂θ avariancereductiontechniqueknownascommonrandom
|     |     |     |     |     |     |     | numbers(Owen,2013). |     | Inpreviouswork, |     |     | whereaclosed |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --------------- | --- | --- | ------------ | --- |
formcomplexitycostorclosedformentropytermareused,
Proof.
|     |     |     |     |     |     |     | part of the | cost is | sensitive  | to particular |               | draws from | the   |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ---------- | ------------- | ------------- | ---------- | ----- |
|     |     |     |     |     |     |     | posterior,  | whilst  | the closed | form part     | is oblivious. |            | Since |
(cid:90)
| ∂   |     | ∂   |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E [f(w,θ)]= f(w,θ)q(w|θ)dw each additive term in the approximate cost in (2) uses the
| ∂θ q(w|θ) |     | ∂θ  |     |     |     |     |     |     |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
sameweightsamples,thegradientsof(2)areonlyaffected
(cid:90)
∂ by the parts of the posterior distribution characterised by
|     |     | =   | f(w,θ)q((cid:15))d(cid:15) |     |     |     |            |          |              |     |     |          |         |
| --- | --- | --- | -------------------------- | --- | --- | --- | ---------- | -------- | ------------ | --- | --- | -------- | ------- |
|     |     | ∂θ  |                            |     |     |     | the weight | samples. | In practice, | we  | did | not find | this to |
(cid:20) ∂f(w,θ)∂w ∂f(w,θ) (cid:21) performbetterthanusingaclosedformKL(whereitcould
=E
|     |     | q((cid:15)) |     |     | +   |     |              |                                  |     |     |     |     |     |
| --- | --- | ----------- | --- | --- | --- | --- | ------------ | -------------------------------- | --- | --- | --- | --- | --- |
|     |     |             |     | ∂w  | ∂θ  | ∂θ  | becomputed), | butwedidnotfindittoperformworse. |     |     |     |     | In  |
ourexperiments,wefoundthatapriorwithoutaneasy-to-
computeclosedformcomplexitycostperformedbest.

WeightUncertaintyinNeuralNetworks
3.2.Gaussianvariationalposterior cross-validationwherepossible. Empiricallywefoundop-
timisingtheparametersofapriorP(w)(bytakingderiva-
| Suppose            | that the | variational | posterior |        | is a diagonal | Gaus-    |          |         |     |            |     |       |                |     |
| ------------------ | -------- | ----------- | --------- | ------ | ------------- | -------- | -------- | ------- | --- | ---------- | --- | ----- | -------------- | --- |
|                    |          |             |           |        |               |          | tives of | (1)) to | not | be useful, | and | yield | worse results. |     |
| sian distribution, |          | then a      | sample    | of the | weights       | w can be |          |         |     |            |     |       |                |     |
Graves(2011)andTitsiasandLa´zaro-Gredilla(2014)pro-
obtainedbysamplingaunitGaussian,shiftingitbyamean
|                                   |           |           |     |      |                 |     | pose closed | form      | updates | of        | the prior | hyperparameters. |             |      |
| --------------------------------- | --------- | --------- | --- | ---- | --------------- | --- | ----------- | --------- | ------- | --------- | --------- | ---------------- | ----------- | ---- |
| µandscalingbyastandarddeviationσ. |           |           |     |      | Weparameterise  |     |             |           |         |           |           |                  |             |      |
|                                   |           |           |     |      |                 |     | Changing    | the prior | based   | upon      | the data  | that             | it is meant | to   |
| the standard                      | deviation | pointwise |     | as σ | = log(1+exp(ρ)) |     |             |           |         |           |           |                  |             |      |
|                                   |           |           |     |      |                 |     | regularise  | is known  | as      | empirical | Bayes     | and              | there is    | much |
andsoσ
|                | isalwaysnon-negative. |         |                           | Thevariationalposterior |     |     |           |        |             |          |                |     |            |      |
| -------------- | --------------------- | ------- | ------------------------- | ----------------------- | --- | --- | --------- | ------ | ----------- | -------- | -------------- | --- | ---------- | ---- |
|                |                       |         |                           |                         |     |     | debate as | to its | validity    | (Gelman, | 2008).         | A   | reason why | it   |
| parametersareθ |                       | =(µ,ρ). | Thusthetransformfromasam- |                         |     |     |           |        |             |          |                |     |            |      |
|                |                       |         |                           |                         |     |     | fails for | Bayes  | by Backprop |          | is as follows: |     | it can be  | eas- |
pleofparameter-freenoiseandthevariationalposteriorpa-
iertochangethepriorparameters(ofwhichtherearefew)
rametersthatyieldsaposteriorsampleoftheweightswis:
thanitistochangetheposteriorparameters(ofwhichthere
| w = t(θ,(cid:15))   | = µ+log(1+exp(ρ))◦(cid:15) |                                  |     |     | where | ◦ is point- |           |     |         |         |           |            |     |        |
| ------------------- | -------------------------- | -------------------------------- | --- | --- | ----- | ----------- | --------- | --- | ------- | ------- | --------- | ---------- | --- | ------ |
|                     |                            |                                  |     |     |       |             | are many) | and | so very | quickly | the prior | parameters |     | try to |
| wisemultiplication. |                            | Eachstepofoptimisationproceedsas |     |     |       |             |           |     |         |         |           |            |     |        |
capturetheempiricaldistributionoftheweightsatthebe-
follows:
|     |     |     |     |     |     |     | ginningoflearning. |          | Thusthepriorlearnstofitpoorinitial |       |          |        |              |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | -------- | ---------------------------------- | ----- | -------- | ------ | ------------ | --- |
|     |     |     |     |     |     |     | parameters         | quickly, | and                                | makes | the cost | in (1) | less willing |     |
1. Sample(cid:15)∼N(0,I).
|     |     |     |     |     |     |     | tomoveawayfrompoorinitialparameters. |     |     |     |     |     | Thiscanyield |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- | --- | ------------ | --- |
Letw=µ+log(1+exp(ρ))◦(cid:15).
| 2.      |         |     |     |     |     |     | slow convergence,      |     | introduce |     | strange | local minima | and | re- |
| ------- | ------- | --- | --- | --- | --- | --- | ---------------------- | --- | --------- | --- | ------- | ------------ | --- | --- |
| 3. Letθ | =(µ,ρ). |     |     |     |     |     | sultinpoorperformance. |     |           |     |         |              |     |     |
4. Letf(w,θ)=logq(w|θ)−logP(w)P(D|w).
|     |     |     |     |     |     |     | We propose | using | a scale | mixture | of  | two Gaussian |     | densi- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | ------- | ------- | --- | ------------ | --- | ------ |
5. Calculatethegradientwithrespecttothemean
|                                                    |     |         |     |         |     |       | ties as the | prior.   | Each   | density         | is zero | mean, | but differing |     |
| -------------------------------------------------- | --- | ------- | --- | ------- | --- | ----- | ----------- | -------- | ------ | --------------- | ------- | ----- | ------------- | --- |
|                                                    |     | ∂f(w,θ) |     | ∂f(w,θ) |     |       | variances:  |          |        |                 |         |       |               |     |
|                                                    | ∆   | µ =     |     | +       |     | . (3) |             |          |        |                 |         |       |               |     |
|                                                    |     |         | ∂w  |         | ∂µ  |       |             |          |        |                 |         |       |               |     |
| 6. Calculatethegradientwithrespecttothestandardde- |     |         |     |         |     |       |             | (cid:89) |        |                 |         |       |               |     |
|                                                    |     |         |     |         |     |       |             |          |        | |0,σ2)+(1−π)N(w |         |       | |0,σ2),       |     |
| viationparameterρ                                  |     |         |     |         |     |       | P(w)=       |          | πN(w j | 1               |         |       | j 2           | (7) |
j
|                                    | ∂f(w,θ) |     | (cid:15)  |     | ∂f(w,θ) |     |        |                             |     |     |     |     |             |     |
| ---------------------------------- | ------- | --- | --------- | --- | ------- | --- | ------ | --------------------------- | --- | --- | --- | --- | ----------- | --- |
|                                    | ∆ =     |     |           |     | +       | .   |        |                             |     |     |     |     |             |     |
|                                    | ρ       |     |           |     |         | (4) |        |                             |     |     |     |     |             |     |
|                                    |         | ∂w  | 1+exp(−ρ) |     |         | ∂ρ  |        |                             |     |     |     |     |             |     |
|                                    |         |     |           |     |         |     | wherew | isthejthweightofthenetwork, |     |     |     |     | N(x|µ,σ2)is |     |
| 7. Updatethevariationalparameters: |         |     |           |     |         |     |        | j                           |     |     |     |     |             |     |
theGaussiandensityevaluatedatxwithmeanµandvari-
µ←µ−α∆ (5) ance σ2 and σ 2 and σ 2 are the variances of the mixture
|     |     |     |     | µ   |     |     |     |     | 1   | 2   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ρ←ρ−α∆ . (6) components. The first mixture component of the prior is
ρ
|     |     |     |     |     |     |     | given a | larger | variance | than | the second, | σ   | > σ , provid- |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | -------- | ---- | ----------- | --- | ------------- | --- |
|     |     |     |     |     |     |     |         |        |          |      |             | 1   | 2             |     |
ingaheaviertailinthepriordensitythanaplainGaussian
∂f(w,θ)
| Notethatthe |     | termofthegradientsforthemeanand |     |     |     |     |                                                   |     |     |     |     |     |     |     |
| ----------- | --- | ------------------------------- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|             | ∂w  |                                 |     |     |     |     | prior. Thesecondmixturecomponenthasasmallvariance |     |     |     |     |     |     |     |
standarddeviationaresharedandareexactlythegradients
σ (cid:28)1causingmanyoftheweightstoaprioritightlycon-
| foundbytheusualbackpropagationalgorithmonaneural |     |     |     |     |     |     | 2                   |     |     |                                  |     |     |     |     |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | -------------------------------- | --- | --- | --- | --- |
|                                                  |     |     |     |     |     |     | centratearoundzero. |     |     | Ourpriorresemblesaspike-and-slab |     |     |     |     |
network. Thus,remarkably,tolearnboththemeanandthe
prior(MitchellandBeauchamp,1988;GeorgeandMcCul-
standarddeviationwemustsimplycalculatetheusualgra-
loch,1993;Chipman,1996),whereinsteadallthepriorpa-
| dients found | by backpropagation, |     |     | and | then scale | and shift |                                      |     |     |     |     |              |     |     |
| ------------ | ------------------- | --- | --- | --- | ---------- | --------- | ------------------------------------ | --- | --- | --- | --- | ------------ | --- | --- |
|              |                     |     |     |     |            |           | rametersaresharedamongalltheweights. |     |     |     |     | Thismakesthe |     |     |
themasabove.
priormoreamenabletouseduringoptimisationbystochas-
ticgradientdescentandavoidstheneedforpriorparameter
3.3.Scalemixtureprior
optimisationbasedupontrainingdata.
HavingliberatedouralgorithmfromtheconfinesofGaus-
sianpriorsandposteriors,weproposeasimplescalemix- 3.4.MinibatchesandKLre-weighting
| ture prior   | combined | with | a diagonal |            | Gaussian | posterior. |            |         |      |        |          |        |             |     |
| ------------ | -------- | ---- | ---------- | ---------- | -------- | ---------- | ---------- | ------- | ---- | ------ | -------- | ------ | ----------- | --- |
|              |          |      |            |            |          |            | As several | authors | have | noted, | the cost | in (1) | is amenable |     |
| The diagonal | Gaussian |      | posterior  | is largely | free     | from nu-   |            |         |      |        |          |        |             |     |
tominibatchoptimisation,oftenusedwithneuralnetworks:
mericalissues,andtwodegreesoffreedomperweightonly
|     |     |     |     |     |     |     | for each | epoch | of optimisation |     | the training |     | data D is | ran- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | --------------- | --- | ------------ | --- | --------- | ---- |
increasesthenumberofparameterstooptimisebyafactor
|            |               |      |        |     |              |        | domly split | into   | a partition |              | of M | equally-sized | subsets, |      |
| ---------- | ------------- | ---- | ------ | --- | ------------ | ------ | ----------- | ------ | ----------- | ------------ | ---- | ------------- | -------- | ---- |
| of two,    | whilst giving | each | weight | its | own quantity | of un- |             |        |             |              |      |               |          |      |
|            |               |      |        |     |              |        | D ,D ,...,D |        | . Each      | gradient     | is   | averaged      | over all | ele- |
| certainty. |               |      |        |     |              |        | 1 2         |        | M           |              |      |               |          |      |
|            |               |      |        |     |              |        | ments in    | one of | these       | minibatches; |      | a trade-off   | between  | a    |
We pick a fixed-form prior and do not adjust its hyper- fullybatchedgradientdescentandafullystochasticgradi-
parameters during training, instead picking the them by entdescent. Graves(2011)proposesminimisingthemini-

WeightUncertaintyinNeuralNetworks
batchcostforminibatchi=1,2,...,M: time,theagentcanunder-explore,asitmaymissmorere-
wardingactions.1
1
F i EQ(D i ,θ)= M KL[q(w|θ)||P(w)] Thompsonsampling(Thompson,1933)isapopularmeans
−E [logP(D |w)]. (8) of picking an action that trades-off between exploitation
q(w|θ) i (picking the best known action) and exploration (picking
Thisisequivalenttothecostin(1)since (cid:80) FEQ(D ,θ)= what might be a suboptimal arm to learn more). Thomp-
i i i
sonsamplingusuallynecessitatesaBayesiantreatmentof
F(D,θ). There are many ways to weight the complexity
the model parameters. At each step, Thompson sampling
costrelativetothelikelihoodcostoneachminibatch. For
draws a new set of parameters and then picks the action
example, if minibatches are partitioned uniformly at ran-
relative to those parameters. This can be seen as a kind
dom,theKLcostcanbedistributednon-uniformlyamong
the minibatches at each epoch. Let π ∈ [0,1]M and of stochastic hypothesis testing: more probable parame-
(cid:80)M
π =1,anddefine:
ters are drawn more often and thus refuted or confirmed
i=1 i thefastest. MoreconcretelyThompsonsamplingproceeds
Fπ(D ,θ)=π KL[q(w|θ)||P(w)] asfollows:
i i i
−E [logP(D |w)] (9)
q(w|θ) i
1. Sampleanewsetofparametersforthemodel.
ThenE [ (cid:80)M Fπ(D ,θ)]=F(D,θ)whereE denotes 2. Pick the action with the highest expected reward ac-
M i=1 i i M
anexpectationovertherandompartitioningofminibatches. cordingtothesampledparameters.
In particular, we found the scheme π = 2M−i to work 3. Updatethemodel. Goto1.
i 2M−1
well: the first few minibatches are heavily influenced by
thecomplexitycost,whilstthelaterminibatchesarelargely Thereisanincreasingliteratureconcerningtheefficacyand
influencedbythedata. Atthebeginningoflearningthisis justificationofthismeansofexploration(ChapelleandLi,
particularlyusefulasforthefirstfewminibatcheschanges 2011; May et al., 2012; Kaufmann et al., 2012; Agrawal
in the weights due to the data are slight and as more data and Goyal, 2012; 2013). Thompson sampling is easily
are seen, data become more influential and the prior less adapted to neural networks using the variational posterior
influential. foundinSection3:
4.ContextualBandits 1. Sample weights from the variational posterior: w ∼
q(w|θ).
Contextualbanditsaresimplereinforcementlearningprob- 2. Receivethecontextx.
lemswithoutpersistentstate(Lietal.,2010;Filippietal., 3. PicktheactionathatminimisesE [r]
P(r|x,a,w)
2010). At each step an agent is presented with a context
4. Receiverewardr.
x and a choice of one of K possible actions a. Different
5. Update variational parameters θ according to Sec-
actionsyielddifferentunknownrewardsr. Theagentmust
tion3. Goto1.
picktheactionthatyieldsthehighestexpectedreward.The
contextisassumedtobepresentedindependentofanypre-
Notethatitispossible,asmentionedinSection3.1,tode-
viousactions,rewardsorcontexts.
creasethevarianceofthegradientestimates,tradingofffor
Anagentbuildsamodelofthedistributionoftherewards reducedexploration,byusingmorethanoneMonteCarlo
conditionedupontheactionandthecontext: P(r|x,a,w). sample, usingthecorrespondingnetworksasanensemble
Itthenusesthismodeltopickitsaction.Note,importantly, and picking the action by minimising the average of the
thatanagentdoesnotknowwhatrewarditcouldhavere- expectations.
ceived for an action that it did not pick, a difficulty often
Initiallythevariationalposteriorwillbeclosetotheprior,
known as “the absence of counterfactual”. As the agent’s
model P(r|x,a,w) is trained online, based upon the ac- andactionswillbepickeduniformly.Astheagenttakesac-
tions, the variationalposteriorwill beginto converge, and
tionschosen,unlessexploratoryactionsaretaken,theagent
uncertainty on many parameters can decrease, and so ac-
mayperformsuboptimally.
tionselectionwillbecomemoredeterministic,focusingon
the high expected reward actions discovered so far. It is
4.1.ThompsonSamplingforNeuralNetworks
AsinSection2,P(r|x,a,w)canbemodelledbyaneural
1Interestingly,dependinguponhowwareinitialisedandthe
meanofpriorusedduringMAPinference, itissometimespos-
network where w are the weights of the neural network.
sible to obtain another heuristic for the exploration-exploitation
However if this network is simply fit to observations and trade-off: optimism-under-uncertainty. We leave this for future
the action with the highest expected reward taken at each investigation.

WeightUncertaintyinNeuralNetworks
Table1.ClassificationErrorRatesonMNIST.(cid:63)indicatesresult
usedanensembleof5networks.
Method
reyaL/stinU#
sthgieW#
2.0
1.6
1.2
Test
Error
0.8
SGD,noregularisation(Simardetal.,2003) 800 1.3m 1.6% 0 100 200 300 400 500 600
Epochs
SGD,dropout(Hintonetal.,2012) ≈1.3%
SGD,dropconnect(Wanetal.,2013) 800 1.3m 1.2%(cid:63)
SGD 400 500k 1.83%
800 1.3m 1.84%
1200 2.4m 1.88%
SGD,dropout 400 500k 1.51%
800 1.3m 1.33%
1200 2.4m 1.36%
BayesbyBackprop,Gaussian 400 500k 1.82%
800 1.3m 1.99%
1200 2.4m 2.04%
BayesbyBackprop,Scalemixture 400 500k 1.36%
800 1.3m 1.34%
1200 2.4m 1.32%
knownthatvariationalmethodsunder-estimateuncertainty
(Minka, 2001; 2005; Bishop, 2006) which could lead to
under-exploration and premature convergence in practice,
butwedidnotfindthisinpractice.
5.Experiments
Wepresentsomeempiricalevaluationofthemethodspro-
posedabove: onMNISTclassification,onanon-linearre-
gressiontask,andonacontextualbanditstask.
5.1.ClassificationonMNIST
We trained networks of various sizes on the MNIST dig-
itsdataset(LeCunandCortes,1998),consistingof60,000
training and 10,000 testing pixel images of size 28 by 28.
Eachimageislabelledwithitscorrespondingnumber(be-
tweenzeroandnine,inclusive). Wepreprocessedthepix-
els by dividing values by 126. Many methods have been
proposed to improve results on MNIST: generative pre-
training,convolutions,distortions,etc. Hereweshallfocus
onimprovingtheperformanceofanordinaryfeedforward
neural network without using any of these methods. We
usedanetworkoftwohiddenlayersofrectifiedlinearunits
(NairandHinton,2010;Glorotetal.,2011),andasoftmax
outputlayerwith10units,oneforeachpossiblelabel.
AccordingtoHintonetal.(2012),thebestpublishedfeed-
forwardneuralnetworkclassificationresultonMNIST(ex-
cluding those using data set augmentation, convolutions,
etc.) is 1.6% (Simard et al., 2003), whilst dropout with
anL2regulariserattainserrorsaround1.3%. Resultsfrom
BayesbyBackpropareshowninTable1,forvarioussized
)%(
rorre
tseT
Algorithm
Bayes by Backprop
Dropout
Vanilla SGD
Figure2. TesterroronMNISTastrainingprogresses.
15
10
5
0
−0.2 −0.1 0.0 0.1 0.2
Weight
ytisneD
Algorithm
Bayes by Backprop
Dropout
Vanilla SGD
Figure3.Histogramofthetrainedweightsoftheneuralnetwork,
forDropout,plainSGD,andsamplesfromBayesbyBackprop.
networks, using either a Gaussian or Gaussian scale mix-
ture prior. Performance is comparable to that of dropout,
perhaps slightly better, as also see on Figure 2. Note that
wetrainedon50,000digitsandused10,000digitsasaval-
idation set, whilst Hinton et al. (2012) trained on 60,000
digits and did not use a validation set. We used the vali-
dation set to pick the best hyperparameters (learning rate,
number of gradients to average) and so we also repeated
thisprotocolfordropoutandSGD(StochasticGradientDe-
scent on the MLE objective in Section 2). We considered
learning rates of 10−3, 10−4 and 10−5 with minibatches
ofsize128. ForBayesbyBackprop,weaveragedoverei-
ther 1, 2, 5, or 10 samples and considered π ∈ {1,1,3},
4 2 4
−logσ ∈{0,1,2}and−logσ ∈{6,7,8}.
1 2
Figure2showsthelearningcurvesonthetestsetforBayes
byBackprop,dropoutandSGDonanetworkwithtwolay-
ers of 1200 rectified linear units. As can be seen, SGD
converges the quickest, initially obtaining a low test er-
ror and then overfitting. Bayes by Backprop and dropout
convergeatsimilarrates(althougheachiterationofBayes
byBackpropismoreexpensivethandropout–aroundtwo
times slower). Eventually Bayes by Backprop converges
onabettertesterrorthandropoutafter600epochs.
Figure3showsdensityestimatesoftheweights.TheBayes
byBackpropweightsaresampledfromthevariationalpos-
terior,andthedropoutweightsarethoseusedattesttime.
Interestingly the regularised networks found by dropout

WeightUncertaintyinNeuralNetworks
andBayesbyBackprophaveagreaterrangeandwithfewer
centredatzerothanthosefoundbySGD.BayesbyBack-
propusesthegreatestrangeofweights.
0.8
0.6
0.4
0.2
0.0
−5.0 −2.5 0.0
Signal−to−Noise Ratio (dB)
ytisneD
1.00
0.75
0.50
0.25
0.00
−7.5 −5.0 −2.5 0.0
Signal−to−Noise Ratio (dB)
FDC
Table2. ClassificationErrorsafterWeightpruning
Proportionremoved #Weights TestError
0% 2.4m 1.24%
50% 1.2m 1.24%
75% 600k 1.24%
95% 120k 1.29%
98% 48k 1.39%
It is interesting to contrast this weight removal approach
to obtaining a fast, smaller, sparse network for prediction
aftertrainingwiththeapproachtakenbydistillation(Hin-
ton et al., 2014) which requires an extra stage of training
to obtain a compressed prediction model. As with distil-
lation, our method begins with an ensemble (one for each
possibleassignmentoftheweights). However,unlikedis-
tillation,wecansimplyobtainasubsetofthisensembleby
usingtheprobabilisticpropertiesoftheweightdistributions
learnttogracefullyprunetheensembledownintoasmaller
network. ThuseventhoughnetworkstrainedbyBayesby
Backpropmayhavetwiceasmanyweights,thenumberof
parametersthatactuallyneedtobestoredatruntimecanbe
far fewer. Graves (2011) also considered pruning weights
usingthesignaltonoiseratio,butdemonstratedresultson
a network 20 times smaller and did not prune as high a
Figure4.Density and CDF of the Signal-to-Noise ratio over all proportionofweights(atmost11%)whilststillmaintain-
weightsinthenetwork.Theredlinedenotesthe75%cut-off.
ing good test performance. The scale mixture prior used
by Bayes by Backprop encourages a broad spread of the
weights.Manyoftheseweightscanbesuccessfullypruned
In Table 2, we examine the effect of replacing the vari-
withoutimpactingperformancesignificantly.
ational posterior on some of the weights with a constant
zero, so as to determine the level of redundancy in the
5.2.Regressioncurves
network found by Bayes by Backprop. We took a Bayes
by Backprop trained network with two layers of 1200 Wegeneratedtrainingdatafromthecurve:
units2 and ordered the weights by their signal-to-noise ra-
tio(|µ |/σ ). Weremovedtheweightswiththelowestsig- y =x+0.3sin(2π(x+(cid:15)))+0.3sin(4π(x+(cid:15)))+(cid:15)
i i
nal to noise ratio. As can be seen in Table 2, even when
95%oftheweightsareremovedthenetworkstillperforms where (cid:15) ∼ N(0,0.02). Figure 5 shows two examples of
well, with a significant drop in performance once 98% of fittinganeuralnetworktothesedata,minimisingacondi-
theweightshavebeenremoved. tional Gaussian loss. Note that in the regions of the input
spacewheretherearenodata,theordinaryneuralnetwork
In Figure 4 we examined the distribution of the signal-to-
reduces the variance to zero and chooses to fit a particu-
noiserelativetothecut-offinthenetworkusesinTable2.
lar function, even though there are many possible extrap-
Thelowerplotshowsthecumulativedistributionofsignal-
olations of the training data. On the left, Bayesian model
to-noiseratio,whilstthetopplotshowsthedensity. From
averagingaffectspredictions: wheretherearenodata,the
thedensityplotweseetherearetwomodalitiesofsignal-
confidence intervals diverge, reflecting there being many
to-noise ratios, and from the CDF we see that the 75%
possible extrapolations. In this case Bayes by Backprop
cut-off separates these two peaks. These two peaks coin-
prefers to be uncertain where there are no nearby data, as
cide with a drop in performance in Table 2 from 1.24%
opposedtoastandardneuralnetworkwhichcanbeoverly
to1.29%,suggestingthatthesignal-to-noiseheuristicisin
confident.
factrelatedtothetestperformance.
2Weusedanetworkfromtheendoftrainingratherthanpick- 5.3.BanditsonMushroomTask
ing a network with a low validation cost found during training,
hencethedisparitywithresultsinTable1. Thelowesttesterror WetaketheUCIMushroomsdataset(BacheandLichman,
observedwas1.12%. 2013), and cast it as a bandit task, similar to Guez (2015,

WeightUncertaintyinNeuralNetworks
| 1.2 |     |                              |     | 1.2 |     |                              |     |                     |        |                                 |     |         |              |     |
| --- | --- | ---------------------------- | --- | --- | --- | ---------------------------- | --- | ------------------- | ------ | ------------------------------- | --- | ------- | ------------ | --- |
|     |     |                              |     |     |     |                              |     | foractionselection. |        | Wekeptthelast4096reward,context |     |         |              |     |
|     |     |                              |     |     |     |                              |     | and action          | tuples | in a buffer,                    | and | trained | the networks | us- |
| 0 . | 8   | xxx x x x x xxxxxxx x xx x x |     | 0 . | 8   | xxx x x x x xxxxxxx x xx x x |     |                     |        |                                 |     |         |              |     |
x xx x xx xx xxxxxxx xxxxxx xx xxxx x xxxxx xx x x xxxxx xxx xx xx xxx x xxxxx xx xx xxxx xx xx xxxxxxxxx xxxxx x xxxxxx x x xxxxxxx x x x xx x xx xx xxxxxxx xxxxxx xx xxxx x xxxxx xx x x xxxxx xxx xx xx xxx x xxxxx xx xx xxxx xx xx xxxxxxxxx xxxxx x xxxxxx x x xxxxxxx x x ingrandomlydrawnminibatchesofsize64for64training
|     |     | xx x xxx x xx x xxx xxx xxxx xx x x x x xx x x x x x x x x x x xx x xx xxxxx xxx xx xx x xx x x x xxxxx x x | xx  |     |     | xx x xxx x xx x xxx xxx xxxx xx x x x x xx x x x x x x x x x x xx x xx xxxxx xxx xx xx x xx x x x xxxxx x x | xx  |     |     |     |     |     |     |     |
| --- | --- | ----------------------------------------------------------------------------------------------------------- | --- | --- | --- | ----------------------------------------------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
x xx x x xxxxxxxx xxxxxxx xx x xx x x x x x xx x x xx xx xxxxxxxx xxxxxx x xxx xx xxx xxx xx x xxxx x xxxx x x x x x xx xxx xx xxxx xxxx x x x x xxx x xxx x xx x x xxxxxxxx xxxxxxx xx x xx x x x x x xx x x xx xx xxxxxxxx xxxxxx x xxx xx xxx xxx xx x xxxx x xxxx x x x x x xx xxx xx xxxx xxxx x x x x xxx x xxx steps(64×64=4096)perinteractionwiththeMushroom
| 0 . | 4   | x x xx xxxx x xx xxx x xx x xx x x xx x x x x xxxx xxx x x xx xxxx x x xxxx x x xxxxxxxx x xxx x x x x xxxxxxxx x x x xxx x xxx x xxxxx x x x xxx | x xxx xxxxxxxxxxxxx xxxxxxxx xx xx xxxxx x x xx x xx x | 0 . | 4   | x x xx xxxx x xx xxx x xx x xx x x xx x x x x xxxx xxx x x xx xxxx x x xxxx x x xxxxxxxx x xxx x x x x xxxxxxxx x x x xxx x xxx x xxxxx | x x x xxx x xxx xxxxxxxxxxxxx xxxxxxxx xx xx xxxxx x x xx x xx x |     |     |     |     |     |     |     |
| --- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | --- | --- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
xxx xxxxxxxxxxxx xxxx x xx x x xx xxx x x xxx x x xx x xx x xx x xxxx x xxxxx xx x x xx xxxxx xxxxxxxxxxx xxx xxx x x x xxx xxx xxx xx x xx x xx xx x x xxx xxxxxxxxxxxx xxxx x xx x x xx xxx x x xxx x x xx x xx x xx x xxxx x xxxxx xx x x xx xxxxx xxxxxxxxxxx xxx xxx x x x xxx xxx xxx xx x xx x xx xx x x bandit. Acommonheuristicfortrading-offexplorationvs.
|     | x x xxx                                    | x xx xx x xx x x x x |     |     | x x xxx                                    | x xx xx x xx x x | x x |     |     |     |     |     |     |     |
| --- | ------------------------------------------ | -------------------- | --- | --- | ------------------------------------------ | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | xxxxx xxxxx xxxxxx x xxx xx xx x x xxxxx x | x                    |     |     | xxxxx xxxxx xxxxxx x xxx xx xx x x xxxxx x | x                |     |     |     |     |     |     |     |     |
0 . 0 xx x x 0 . 0 xx x x exploitation is to follow an ε-greedy policy: with proba-
|     | xx  |     |     |     | xx  |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bilityεproposeauniformlyrandomaction,otherwisepick
| −0.4 |     |     |     | −0.4 |     |     |     |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thebestactionaccordingtotheneuralnetwork.
|     |     |     |     |     |     |     |         | Figure 6        | compares | a Bayes                       | by Backprop |     | agent with | three |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------------- | -------- | ----------------------------- | ----------- | --- | ---------- | ----- |
|     | 0.0 | 0.4 | 0.8 | 1.2 | 0.0 | 0.4 | 0.8 1.2 |                 |          |                               |             |     |            |       |
|     |     |     |     |     |     |     |         | ε-greedyagents, |          | forvaluesofεof0%(puregreedy), |             |     |            | 1%,   |
Figure5.Regressionofnoisydatawithinterquatileranges.Black and 5%. An ε of 5% appears to over-explore, whereas a
crosses are training samples. Red lines are median predictions. purely greedy agent does poorly at the beginning, greed-
| Blue/purpleregionisinterquartilerange. |     |     |     |     |     | Left: BayesbyBack- |     |     |     |     |     |     |     |     |
| -------------------------------------- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
ilyelectingtoeatnothing,butthendoesmuchbetteronce
propneuralnetwork,Right:standardneuralnetwork. it has seen enough data. It seems that non-local function
|     |     |     |     |     |     |     |     | approximation |     | updates | allow the | greedy | agent to | explore, |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------- | --------- | ------ | -------- | -------- |
asforthefirst1,000steps,theagenteatsnothingbutafter
approximately1,000thegreedyagentsuddenlydecidesto
tergeR evitalumuC eat mushrooms. The Bayes by Backprop agent explores
10000
|     |     |     |     |     |     |     |     | from the | beginning, | both | eating | and ignoring | mushrooms |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ---- | ------ | ------------ | --------- | --- |
andquicklyconvergesoneatingandnon-eatingwithanal-
5% Greedy
mostperfectrate(hencethealmostflatregret).
1% Greedy
Greedy
1000
Bayes by Backprop
6.Discussion
|     |     | 0   | 10000 | 20000 | 30000 | 40000 | 50000 |               |             |                 |        |              |              |      |
| --- | --- | --- | ----- | ----- | ----- | ----- | ----- | ------------- | ----------- | --------------- | ------ | ------------ | ------------ | ---- |
|     |     |     |       | Step  |       |       |       | We introduced |             | a new algorithm |        | for learning | neural       | net- |
|     |     |     |       |       |       |       |       | works with    | uncertainty |                 | on the | weights      | called Bayes | by   |
Figure6.Comparison of cumulative regret of various agents on Backprop. It optimises a well-defined objective function
| themushroombandittask,averagedoverfiveruns. |     |     |     |     |     |     | Lowerisbet- |          |                |     |             |     |          |          |
| ------------------------------------------- | --- | --- | --- | --- | --- | --- | ----------- | -------- | -------------- | --- | ----------- | --- | -------- | -------- |
|                                             |     |     |     |     |     |     |             | to learn | a distribution | on  | the weights | of  | a neural | network. |
ter.
|     |     |     |     |     |     |     |     | The algorithm |     | achieves | good results | in  | several domains. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | ------------ | --- | ---------------- | --- |
WhenclassifyingMNISTdigits,performancefromBayes
|     |     |     |     |     |     |     |     | byBackpropiscomparabletothatofdropout. |     |     |     |     | Wedemon- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | -------- | --- |
Chapter6).Eachmushroomhasasetoffeatures,whichwe
|     |     |     |     |     |     |     |     | strated on | a simple | non-linear | regression |     | problem | that the |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ---------- | ---------- | --- | ------- | -------- |
treatasthecontextforthebandit,andislabelledasedible
or poisonous. An agent can either eat or not eat a mush- uncertainty introduced allows the network to make more
|     |     |     |     |     |     |     |     | reasonablepredictionsaboutunseendata. |     |     |     |     | Finally,forcon- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --------------- | --- |
room. Ifanagenteatsanediblemushroom,thenitreceives
|             |             |                                      |             |        |        |         |             | textual       | bandits,                               | we showed | how          | Bayes       | by Backprop | can     |
| ----------- | ----------- | ------------------------------------ | ----------- | ------ | ------ | ------- | ----------- | ------------- | -------------------------------------- | --------- | ------------ | ----------- | ----------- | ------- |
| arewardof5. |             | Ifanagenteatsapoisonousmushroom,then |             |        |        |         |             |               |                                        |           |              |             |             |         |
|             |             |                                      | 1           |        |        |         |             | automatically | learn                                  | how       | to trade-off | exploration |             | and ex- |
| with        | probability |                                      | it receives | a      | reward | of −35, | otherwise   |               |                                        |           |              |             |             |         |
|             |             |                                      | 2           |        |        |         |             | ploitation.   | SinceBayesbyBackpropsimplyusesgradient |           |              |             |             |         |
| a reward    |             | of 5.                                | If an agent | elects | not    | to eat  | a mushroom, |               |                                        |           |              |             |             |         |
itreceivesarewardof0. Thusanagentexpectstoreceive updates,itcanreadilybescaledusingmulti-machineopti-
misationschemessuchasasynchronousSGD(Deanetal.,
| arewardof5foreatinganediblereward, |     |     |     |     |     | butanexpected |     |        |              |     |                   |     |          |         |
| ---------------------------------- | --- | --- | --- | --- | --- | ------------- | --- | ------ | ------------ | --- | ----------------- | --- | -------- | ------- |
|                                    |     |     |     |     |     |               |     | 2012). | Furthermore, | all | of the operations |     | used are | readily |
rewardof−15foreatingapoisonousmushroom.
implementedonaGPU.
Regretmeasuresthedifferencebetweentherewardachiev-
| able | by an | oracle | and the | reward | received | by  | an agent. In |     |     |     |     |     |     |     |
| ---- | ----- | ------ | ------- | ------ | -------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
thiscase,anoraclewillalwaysreceivearewardof5foran
|     |     |     |     |     |     |     |     | Acknowledgements |     | TheauthorswouldliketothankIvo |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ----------------------------- | --- | --- | --- | --- |
ediblemushroom,or0forapoisonousmushroom.Wetake Danihelka, DaniloRezende, SilviaChiappa, AlexGraves,
| the | cumulative |     | sum of | regret of | several | agents | and show |             |     |             |      |         |       |       |
| --- | ---------- | --- | ------ | --------- | ------- | ------ | -------- | ----------- | --- | ----------- | ---- | ------- | ----- | ----- |
|     |            |     |        |           |         |        |          | Remi Munos, |     | Ben Coppin, | Liam | Clancy, | James | Kirk- |
them in Figure 6. Each agent uses a neural network with patrick,ShakirMohamed,DavidPfau,andTheophaneWe-
two hidden layers of 100 rectified linear units. The input berforusefuldiscussionsandcomments.
tothenetworkisavectorconsistingofthemushroomfea-
| tures(context)andaoneofK |     |     |     | encodingoftheaction. |     |     | The |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
References
outputofthenetworkisasinglescalar,representingtheex-
pectedrewardofthegivenactioninthegivencontext. For ShipraAgrawalandNavinGoyal. AnalysisofThompson
BayesbyBackprop,wesampledtheweightstwiceandav- sampling for the multi-armed bandit problem. In Pro-
eraged two of these outputs to obtain the expected reward ceedings of the 25th Annual Conference On Learning

WeightUncertaintyinNeuralNetworks
Theory(COLT),volume23,pages39.1–39.26,2012. AlexGraves. Practicalvariationalinferenceforneuralnet-
|                |     |           |        |         |         |        | works. | In Advances |     | in Neural | Information |     | Processing |     |
| -------------- | --- | --------- | ------ | ------- | ------- | ------ | ------ | ----------- | --- | --------- | ----------- | --- | ---------- | --- |
| Shipra Agrawal |     | and Navin | Goyal. | Further | optimal | regret |        |             |     |           |             |     |            |     |
Systems(NIPS),pages2348–2356,2011.
| bounds | for Thompson |     | sampling. | In Proceedings |     | of the |     |     |     |     |     |     |     |     |
| ------ | ------------ | --- | --------- | -------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
16th International Conference on Artificial Intelligence KarolGregor,IvoDanihelka,AndriyMnih,CharlesBlun-
andStatisticsLearning(AISTATS),pages99–107,2013. dell, and Daan Wierstra. Deep AutoRegressive net-
|     |     |     |     |     |     |     | works. | InProceedingsofthe31stInternationalConfer- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
UCIMachineLearning
KevinBacheandMosheLichman. ence on Machine Learning (ICML), pages 1242–1250,
| Repository. | University |     | of California, |     | Irvine, | School of | 2014. |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | -------------- | --- | ------- | --------- | ----- | --- | --- | --- | --- | --- | --- | --- |
InformationandComputerSciences,2013.URLhttp:
|     |     |     |     |     |     |     | Arthur Guez. |     | Sample-Based |     | Search | Methods | For | Bayes- |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------------ | --- | ------ | ------- | --- | ------ |
//archive.ics.uci.edu/ml.
|                     |     |                                   |     |     |     |     | AdaptivePlanning. |     | PhDthesis,UniversityCollegeLon- |     |     |     |     |     |
| ------------------- | --- | --------------------------------- | --- | --- | --- | --- | ----------------- | --- | ------------------------------- | --- | --- | --- | --- | --- |
| ChristopherMBishop. |     | Section10.1:variationalinference. |     |     |     |     | don,2015.         |     |                                 |     |     |     |     |     |
InPatternRecognitionandMachineLearning.Springer,
|     |     |     |     |     |     |     | Geoffrey | Hinton, | Oriol | Vinyals, | and | Jeff Dean. | Distilling |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------- | ----- | -------- | --- | ---------- | ---------- | --- |
2006. ISBN9780387310732.
|     |     |     |     |     |     |     | theknowledgeinaneuralnetwork. |     |     |     |     | InNIPS2014Deep |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | -------------- | --- | --- |
WrayLBuntineandAndreasSWeigend. Bayesianback- LearningandRepresentationLearningWorkshop,2014.
| propagation. | Complexsystems,5(6):603–643,1991. |     |     |     |     |     |                                |     |     |     |     |                |     |     |
| ------------ | --------------------------------- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --- | -------------- | --- | --- |
|              |                                   |     |     |     |     |     | GeoffreyEHintonandDrewVanCamp. |     |     |     |     | Keepingtheneu- |     |     |
ralnetworkssimplebyminimizingthedescriptionlength
OlivierChapelleandLihongLi.Anempiricalevaluationof
InProceedingsofthe16thAnnualCon-
| Thompsonsampling.InAdvancesinNeuralInformation |     |     |     |     |     |     | oftheweights. |     |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
ProcessingSystems(NIPS),pages2249–2257,2011. ferenceOnLearningTheory(COLT),pages5–13.ACM,
1993.
| Hugh Chipman. |                                          | Bayesian | variable | selection    | with    | related   |                                         |                           |        |             |               |      |             |         |
| ------------- | ---------------------------------------- | -------- | -------- | ------------ | ------- | --------- | --------------------------------------- | ------------------------- | ------ | ----------- | ------------- | ---- | ----------- | ------- |
|               |                                          |          |          |              |         |           | Geoffrey                                | E. Hinton,                | Nitish | Srivastava, |               | Alex | Krizhevsky, |         |
| predictors.   | CanadianJournalofStatistics,24(1):17–36, |          |          |              |         |           |                                         |                           |        |             |               |      |             |         |
| 1996.         |                                          |          |          |              |         |           | IlyaSutskever,andRuslanR.Salakhutdinov. |                           |        |             |               |      | Improving   |         |
|               |                                          |          |          |              |         |           | neural                                  | networks                  | by     | preventing  | co-adaptation |      | of          | feature |
|               |                                          |          |          |              |         |           | detectors.                              | arXiv:1207.0580,July2012. |        |             |               |      |             |         |
| Jeffrey Dean, | Greg                                     | Corrado, |          | Rajat Monga, |         | Kai Chen, |                                         |                           |        |             |               |      |             |         |
| Matthieu      | Devin,                                   | Mark     | Mao,     | Andrew       | Senior, | Paul      |                                         |                           |        |             |               |      |             |         |
TommiS.JaakkolaandMichaelI.Jordan.Bayesianparam-
| Tucker,  | Ke Yang, | Quoc      | V Le, | et al.   | Large     | scale dis- |                 |     |                 |     |          |     |            |     |
| -------- | -------- | --------- | ----- | -------- | --------- | ---------- | --------------- | --- | --------------- | --- | -------- | --- | ---------- | --- |
|          |          |           |       |          |           |            | eter estimation |     | via variational |     | methods. |     | Statistics | and |
| tributed | deep     | networks. | In    | Advances | in Neural | Infor-     |                 |     |                 |     |          |     |            |     |
Computing,10(1):25–37,2000.
| mation         | Processing | Systems |         | (NIPS),   | pages 1223–1231, |           |                  |           |           |                   |        |     |         |         |
| -------------- | ---------- | ------- | ------- | --------- | ---------------- | --------- | ---------------- | --------- | --------- | ----------------- | ------ | --- | ------- | ------- |
| 2012.          |            |         |         |           |                  |           | Emilie Kaufmann, |           | Nathaniel |                   | Korda, | and | Re´mi   | Munos.  |
|                |            |         |         |           |                  |           | Thompson         | sampling: |           | An asymptotically |        |     | optimal | finite- |
| Sarah Filippi, | Olivier    | Cappe,  | Aurlien | Garivier, |                  | and Csaba |                  |           |           |                   |        |     |         |         |
timeanalysis.InProceedingsofthe23rdAnnualConfer-
| Szepesvri. | Parametric |     | bandits: | The | generalized | linear |     |     |     |     |     |     |     |     |
| ---------- | ---------- | --- | -------- | --- | ----------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
enceonAlgorithmicLearningTheory(ALT),pages199–
case.InAdvancesinNeuralInformationProcessingSys-
213.Springer,2012.
tems,pages586–594,2010.
DiederikP.KingmaandMaxWelling.Auto-encodingvari-
| Karl Friston,   | Je´re´mie   | Mattout, |                | Nelson      | Trujillo-Barreto, |          |            |        |                |                        |     |         |               |       |
| --------------- | ----------- | -------- | -------------- | ----------- | ----------------- | -------- | ---------- | ------ | -------------- | ---------------------- | --- | ------- | ------------- | ----- |
|                 |             |          |                |             |                   |          | ational    | Bayes. | In Proceedings |                        | of  | the 2nd | International |       |
| John Ashburner, |             | and      | Will Penny.    | Variational |                   | free en- |            |        |                |                        |     |         |               |       |
|                 |             |          |                |             |                   |          | Conference |        | onLearning     | Representations(ICLR), |     |         |               | 2014. |
| ergy and        | the Laplace |          | approximation. |             | Neuroimage,       | 34       |            |        |                |                        |     |         |               |       |
arXiv: 1312.6114.
(1):220–234,2007.
|     |     |     |     |     |     |     | Yann LeCun. |     | Une proce´dure |     | d’apprentissage |     | pour | re´seau |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | -------------- | --- | --------------- | --- | ---- | ------- |
Andrew Gelman. Objections to Bayesian statistics. a`seuilasymmetrique(alearningschemeforasymmetric
| BayesianAnalysis,3:445–450,2008. |     |     |     |     | ISSN1931-6690. |     |                                 |            |     |                |     |     |           |     |
| -------------------------------- | --- | --- | --- | --- | -------------- | --- | ------------------------------- | ---------- | --- | -------------- | --- | --- | --------- | --- |
|                                  |     |     |     |     |                |     | threshold                       | networks). |     | In Proceedings |     | of  | Cognitiva | 85, |
| doi: 11.1214/08-BA318.           |     |     |     |     |                |     | Paris,France,pages599–604,1985. |            |     |                |     |     |           |     |
EdwardIGeorgeandRobertEMcCulloch. Variableselec- Yann LeCun and Corinna Cortes. The MNIST database
tionviagibbssampling. JournaloftheAmericanStatis- of handwritten digits. http://yann.
|     |     |     |     |     |     |     |     |     |     | 1998. | URL |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- |
ticalAssociation,88(423):881–889,1993. lecun.com/exdb/mnist/.
XavierGlorot,AntoineBordes,andYoshuaBengio. Deep Lihong Li, Wei Chu, John Langford, and Robert E.
sparse rectifier networks. In Proceedings of the 14th Schapire. A contextual-bandit approach to personal-
International Conference on Artificial Intelligence and ized news article recommendation. In Proceedings of
Statistics Learning (AISTATS), volume 15, pages 315– the 19th International Conference on World Wide Web,
| 323,2011. |     |     |     |     |     |     | WWW | ’10, | pages | 661–670, | New | York, | NY, | USA, |
| --------- | --- | --- | --- | --- | --- | --- | --- | ---- | ----- | -------- | --- | ----- | --- | ---- |

WeightUncertaintyinNeuralNetworks
2010. ACM. ISBN 978-1-60558-799-8. doi: 10.1145/ PatriceYSimard,DaveSteinkraus,andJohnCPlatt. Best
1772690.1772758. practicesforconvolutionalneuralnetworksappliedtovi-
sualdocumentanalysis.InProceedingsofthe12thInter-
| David JC | MacKay. | A practical | Bayesian | framework |     | for |     |     |     |     |
| -------- | ------- | ----------- | -------- | --------- | --- | --- | --- | --- | --- | --- |
nationalConferenceonDocumentAnalysisandRecog-
backpropagation networks. Neural computation, 4(3): nition (ICDAR), volume 2, pages 958–958. IEEE Com-
448–472,1992.
puterSociety,2003.
| David JC | MacKay. | Probable | networks |     | and plausible |     |                   |                               |     |     |
| -------- | ------- | -------- | -------- | --- | ------------- | --- | ----------------- | ----------------------------- | --- | --- |
|          |         |          |          |     |               |     | WilliamRThompson. | Onthelikelihoodthatoneunknown |     |     |
predictions-a review of practical Bayesian methods for probability exceeds another in view of the evidence of
| supervised | neural | networks. | Network: | Computation |     | in  |             |                               |     |     |
| ---------- | ------ | --------- | -------- | ----------- | --- | --- | ----------- | ----------------------------- | --- | --- |
|            |        |           |          |             |     |     | twosamples. | Biometrika,pages285–294,1933. |     |     |
NeuralSystems,6(3):469–505,1995.
|     |     |     |     |     |     |     | Michalis Titsias | and Miguel | La´zaro-Gredilla. | Doubly |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ---------- | ----------------- | ------ |
Benedict C May, Nathan Korda, Anthony Lee, and stochasticvariationalbayesfornon-conjugateinference.
David S. Leslie. Optimistic Bayesian sampling in In Proceedings of the 31st International Conference on
| contextual-bandit |     | problems. | The | Journal | of Machine |     |     |     |     |     |
| ----------------- | --- | --------- | --- | ------- | ---------- | --- | --- | --- | --- | --- |
MachineLearning(ICML-14),pages1971–1979,2014.
LearningResearch,13(1):2069–2106,2012.
|     |     |     |     |     |     |     | Li Wan, Matthew | Zeiler, | Sixin Zhang, Yann | L Cun, and |
| --- | --- | --- | --- | --- | --- | --- | --------------- | ------- | ----------------- | ---------- |
ThomasPMinka. Afamilyofalgorithmsforapproximate Rob Fergus. Regularization of neural networks us-
Bayesianinference. PhDthesis,MassachusettsInstitute ing dropconnect. In Proceedings of the 30th Inter-
ofTechnology,2001. national Conference on Machine Learning (ICML-13),
pages1058–1066,2013.
| ThomasPMinka. |     | Divergencemeasuresandmessagepass- |     |     |     |     |     |     |     |     |
| ------------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
ing. Technicalreport,MicrosoftResearch,2005. Jonathan S Yedidia, William T Freeman, and Yair Weiss.
|                                 |     |     |     |                  |     |     | Generalized | belief propagation. | In Advances | in Neu- |
| ------------------------------- | --- | --- | --- | ---------------- | --- | --- | ----------- | ------------------- | ----------- | ------- |
| TobyJMitchellandJohnJBeauchamp. |     |     |     | Bayesianvariable |     |     |             |                     |             |         |
ralInformationProcessingSystems(NIPS),volume13,
selection in linear regression. Journal of the American pages689–695,2000.
StatisticalAssociation,83(404):1023–1032,1988.
| Vinod Nair | and Geoffrey | E Hinton. | Rectified |     | linear | units |     |     |     |     |
| ---------- | ------------ | --------- | --------- | --- | ------ | ----- | --- | --- | --- | --- |
InProceedings
improverestrictedBoltzmannmachines.
ofthe27thInternationalConferenceonMachineLearn-
ing(ICML),pages807–814,2010.
| RadfordMNealandGeoffreyEHinton. |             |                        |     | AviewoftheEM |       |       |     |     |     |     |
| ------------------------------- | ----------- | ---------------------- | --- | ------------ | ----- | ----- | --- | --- | --- | --- |
| algorithm                       | that        | justifies incremental, |     | sparse,      | and   | other |     |     |     |     |
| variants.                       | In Learning | in graphical           |     | models,      | pages | 355–  |     |     |     |     |
368.Springer,1998.
| Manfred Opper                   | and | Ce´dric Archambeau. |     | The                | variational |     |     |     |     |     |
| ------------------------------- | --- | ------------------- | --- | ------------------ | ----------- | --- | --- | --- | --- | --- |
| Gaussianapproximationrevisited. |     |                     |     | Neuralcomputation, |             |     |     |     |     |     |
21(3):786–792,2009.
| ArtB.Owen. | MonteCarlotheory,methodsandexamples. |     |     |     |     |     |     |     |     |     |
| ---------- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2013.
| Danilo Jimenez                   |            | Rezende, Shakir | Mohamed, |                 | and         | Daan |     |     |     |     |
| -------------------------------- | ---------- | --------------- | -------- | --------------- | ----------- | ---- | --- | --- | --- | --- |
| Wierstra.                        | Stochastic | backpropagation |          | and             | approximate |      |     |     |     |     |
| inferenceindeepgenerativemodels. |            |                 |          | InProceedingsof |             |      |     |     |     |     |
the31stInternationalConferenceonMachineLearning
(ICML),pages1278–1286,2014.
| David E Rumelhart, |     | Geoffrey | E Hinton, |     | and Ronald | J   |     |     |     |     |
| ------------------ | --- | -------- | --------- | --- | ---------- | --- | --- | --- | --- | --- |
Williams.Learningrepresentationsbyback-propagating
errors. Cognitivemodeling,5,1988.
LawrenceKSaul,TommiJaakkola,andMichaelIJordan.
| Meanfieldtheoryforsigmoidbeliefnetworks. |     |     |     |     | Journal |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- |
ofartificialintelligenceresearch,4(1):61–76,1996.