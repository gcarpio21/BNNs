Laplace Redux – Effortless Bayesian Deep Learning
ErikDaxberger∗,c,m AgustinusKristiadi∗,t AlexanderImmer∗,e,p RunaEschenhagen∗,t
MatthiasBauerd PhilippHennigt,m
cUniversityofCambridge
mMPIforIntelligentSystems,Tübingen
tUniversityofTübingen
eDepartmentofComputerScience,ETHZurich
pMaxPlanckETHCenterforLearningSystems
dDeepMind,London
Abstract
Bayesianformulationsofdeeplearninghavebeenshowntohavecompellingtheo-
reticalpropertiesandofferpracticalfunctionalbenefits,suchasimprovedpredictive
uncertaintyquantificationandmodelselection.TheLaplaceapproximation(LA)
isaclassic,andarguablythesimplestfamilyofapproximationsfortheintractable
posteriors of deep neural networks. Yet,despite its simplicity,the LA is not as
popularasalternativeslikevariationalBayesordeepensembles.Thismaybedue
toassumptionsthattheLAisexpensiveduetotheinvolvedHessiancomputation,
thatitisdifficulttoimplement,orthatityieldsinferiorresults.Inthisworkweshow
thatthesearemisconceptions:we(i)reviewtherangeofvariantsoftheLAinclud-
ingversionswithminimalcostoverhead;(ii)introducelaplace,aneasy-to-use
softwarelibraryforPyTorchofferinguser-friendlyaccesstoallmajorflavorsofthe
LA;and(iii)demonstratethroughextensiveexperimentsthattheLAiscompetitive
withmorepopularalternativesintermsofperformance,whileexcellinginterms
ofcomputationalcost.Wehopethatthisworkwillserveasacatalysttoawider
adoptionoftheLAinpracticaldeeplearning,includingindomainswhereBayesian
approachesarenottypicallyconsideredatthemoment.
laplacelibrary:https://github.com/AlexImmer/Laplace
Experiments:https://github.com/runame/laplace-redux
1 Introduction
Despitetheirsuccesses,modernneuralnetworks(NNs)stillsufferfromseveralshortcomingsthat
limit their applicability in some settings. These include (i) poor calibration and overconfidence,
especiallywhenthedatadistributionshiftsbetweentrainingandtesting[1],(ii)catastrophicforgetting
of previously learned tasks when continuously trained on new tasks [2],and (iii) the difficulty of
selectingsuitableNNarchitecturesandhyperparameters[3].Bayesianmodeling[4,5]providesa
principledandunifiedapproachtotackletheseissuesby(i)equippingmodelswithrobustuncertainty
estimates[6],(ii)enablingmodelstolearncontinuallybycapturingpastinformation[7],and(iii)
allowingforautomatedmodelselectionbyoptimallytradingoffdatafitandmodelcomplexity[8].
EventhoughthisprovidescompellingmotivationforusingBayesianneuralnetworks(BNNs)[9],
theyhavenotgainedmuchtractioninpractice.CommoncriticismsincludethatBNNsaredifficult
∗Equal contributors; author ordering sampled uniformly at random. Correspondence to:
ead54@cam.ac.uk, agustinus.kristiadi@uni-tuebingen.de, alexander.immer@inf.ethz.ch,
runa.eschenhagen@student.uni-tuebingen.de.
35thConferenceonNeuralInformationProcessingSystems(NeurIPS2021).
2202
raM
41
]GL.sc[
3v60841.6012:viXra

(a)MAPEstimation (b)LaplaceApproximation (c)Prediction
Figure1:ProbabilisticpredictionswiththeLaplaceapproximationinthreesteps.(a)Wefind
aMAPestimate(yellowstar)viastandardtraining(backgroundcontours=log-posteriorlandscape
on the two-dimensional PCA subspace of the SGD trajectory [30]). (b) We locally approximate
theposteriorlandscapebyfittingaGaussiancenteredattheMAPestimate(yellowcontours),with
covariancematrixequaltothenegativeinverseHessianofthelossattheMAP—thisistheLaplace
approximation(LA).(c)WeusetheLAtomakepredictionswithpredictiveuncertaintyestimates—
here,theblackcurveisthepredictivemean,andtheshadingcoversthe95%confidenceinterval.
toimplement,finickytotune,expensivetotrain,andhardtoscaletomodernmodelsanddatasets.
Forinstance,popularvariationalBayesianmethods[10–12,etc.]requireconsiderablechangestothe
trainingprocedureandmodelarchitecture.Also,theiroptimizationprocessisslowerandtypically
moreunstableunlesscarefullytuned[13].Othermethods,suchasdeepensembles[14],MonteCarlo
dropout[6],andSWAG[15]promisetobringuncertaintyquantificationtostandardNNsinsimple
manners.Butthesemethodseitherrequireasignificantcostincreasecomparedtoasinglenetwork,
havelimitedempiricalperformance,oranunsatisfyingBayesianinterpretation.
Inthispaper,wearguethattheLaplaceapproximation(LA)isasimpleandcost-efficient,yetcompet-
itiveapproximationmethodforinferenceinBayesiandeeplearning.Firstproposedinthiscontextby
MacKay[16],theLAdatesbacktothe18thcentury[17].Itlocallyapproximatestheposteriorwith
aGaussiandistributioncenteredatalocalmaximum,withcovariancematrixcorrespondingtothe
localcurvature.TwokeyadvantagesoftheLAarethatthelocalmaximumisreadilyavailablefrom
standardmaximumaposteriori(MAP)trainingofNNs,andthatcurvatureestimatescanbeeasilyand
efficientlyobtainedthankstorecentadvancesinsecond-orderoptimization,bothintermsofmore
efficientapproximations to the Hessian [18–20] andeasy-to-use software libraries [21]. Together,
theymaketheLApracticalandreadilyapplicabletomanyalready-trainedNNs—theLAessentially
enablespractitionerstoturntheirhigh-performingpoint-estimateNNsintoBNNseasilyandquickly,
withoutlossofpredictiveperformance. Furthermore,theLAtothemarginallikelihoodmayeven
beusedforBayesianmodelselectionorNNtraining[8,22].Figure1providesanintuitionofthe
LA—wefirstfitapointestimateofthemodelandthenestimateaGaussiandistributionaroundthat.
Yet,despite recent progress in scaling and improving the LA for deep learning [23–29],it is far
lesswidespreadthanothermethods.Thisislikelyduetomisconceptions,likethattheLAishardto
implementduetotheHessiancomputation,thatitmustnecessarilyperformworsethanthecompetitors
due to its local nature,orquite simply that it is old and too simple. Here,we show that these are
indeedmisconceptions.Moreover,wearguethattheLAdeservesawideradoptioninbothpractical
andresearch-orienteddeeplearning.Tothisend,ourworkmakesthefollowingcontributions:
1. Wefirstsurveyrecentadvancesandpresentthekeycomponentsofscalableandpractical
Laplaceapproximationsindeeplearning(Section2).
2. Wethenintroducelaplace,aneasy-to-usePyTorch-basedlibraryfor“turningaNNintoa
BNN”viatheLA(Section3).laplaceimplementsawiderangeofdifferentLAvariants.
3. Lastly,usinglaplace,weshowinanextensiveempiricalstudythattheLAiscompetitive
toalternativeapproaches,especiallyconsideringhowsimpleandcheapitis(Section4).
2 TheLaplaceApproximationinDeepLearning
The LA can be usedin two differentways to benefitdeeplearning: Firstly,we can use the LA to
approximatethemodel’sposteriordistribution(seeEq.(5)below)toenableprobabilisticpredictions
(asalsoillustratedinFig.1).Secondly,wecanusetheLAtoapproximatethemodelevidence(see
Eq.(6))toenablemodelselection(e.g.hyperparametertuning).
2

Thecanonicalformof(supervised)deeplearningisthatofempiricalriskminimization.Given,e.g.,an
i.i.d.classificationdatasetD :={(x ∈RM,y ∈RC)}N ,theweightsθ ∈RD ofanL-layerNN
|     |     |     | n   |     | n   | n=1 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
f :RM →RC aretrainedtominimizethe(regularized)empiricalrisk,whichtypicallydecomposes
θ
| intoasumoverempiricallossterms(cid:96)(x |     |     |     | ,y  | ;θ)andaregularizerr(θ), |          |     |     |          |     |
| ---------------------------------------- | --- | --- | --- | --- | ----------------------- | -------- | --- | --- | -------- | --- |
|                                          |     |     |     | n   | n                       |          |     |     |          |     |
|                                          |     |     |     |     |                         | (cid:16) |     |     | (cid:17) |     |
θ =argmin L(D;θ)=argmin r(θ)+ (cid:80)N (cid:96)(x ,y ;θ) . (1)
| MAP |     |     | θ∈RD |     |     | θ∈RD |     | n=1 | n n |     |
| --- | --- | --- | ---- | --- | --- | ---- | --- | --- | --- | --- |
FromtheBayesianviewpoint,thesetermscanbeidentifiedwithi.i.d.log-likelihoodsandalog-prior,
respectivelyand,thus,θ isindeedamaximuma-posteriori(MAP)estimate:
MAP
|     | (cid:96)(x | ,y ;θ)=−logp(y |     | |f  | (x )) | and | r(θ)=−logp(θ) |     |     | (2) |
| --- | ---------- | -------------- | --- | --- | ----- | --- | ------------- | --- | --- | --- |
|     |            | n n            |     | n   | θ n   |     |               |     |     |     |
1γ−2(cid:107)θ(cid:107)2(a.k.a.weightdecay)corresponds
Forexample,thewidelyusedweightregularizerr(θ)=
2
toacenteredGaussianpriorp(θ)=N(θ;0,γ2I),andthecross-entropylossamountstoacategor-
icallikelihood.Hence,theexponentialofthenegativetraininglossexp(−L(D;θ))amountstoan
unnormalizedposterior.Bynormalizingit,weobtain
| p(θ|D)= |     | 1 p(D|θ)p(θ)= |     | 1 exp(−L(D;θ)), |     |     | Z := | (cid:82) p(D|θ)p(θ)dθ |     |     |
| ------- | --- | ------------- | --- | --------------- | --- | --- | ---- | --------------------- | --- | --- |
(3)
|     |     | Z   |     | Z   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
withanintractablenormalizingconstantZ.Laplaceapproximations[17]useasecond-orderexpan-
sionofLaroundθ toconstructaGaussianapproximationtop(θ|D).I.e.weconsider:
MAP
|     |              |     |     |          |     | (cid:124)(cid:0) |     | (cid:1) |     |     |
| --- | ------------ | --- | --- | -------- | --- | ---------------- | --- | ------- | --- | --- |
|     | L(D;θ)≈L(D;θ |     |     | )+ 1(θ−θ |     | ) ∇2L(D;θ)|      |     | (θ−θ    | ),  | (4) |
|     |              |     | MAP | 2        | MAP | θ                | θ   | MAP     | MAP |     |
wherethefirst-ordertermvanishesatθ .ThenwecanidentifytheLaplaceapproximationas
MAP
Laplaceposteriorapproximation
|     |              |     |     |     |      |      | (cid:0)   |       | (cid:1)−1 |     |
| --- | ------------ | --- | --- | --- | ---- | ---- | --------- | ----- | --------- | --- |
|     | p(θ|D)≈N(θ;θ |     | ,Σ) |     | with | Σ := | ∇2L(D;θ)| |       | .         | (5) |
|     |              |     | MAP |     |      |      | θ         | θ MAP |           |     |
ThenormalizingconstantZ (whichistypicallyreferredtoasthemarginallikelihoodorevidence)is
usefulformodelselectionandcanalsobeapproximatedas
Laplaceapproximationoftheevidence
|     |     | Z   | ≈exp(−L(D;θ |     | ))(2π)D/2(detΣ)1/2. |     |     |     |     | (6) |
| --- | --- | --- | ----------- | --- | ------------------- | --- | --- | --- | --- | --- |
MAP
SeeAppendixAformoredetails.Thus,toobtaintheapproximateposterior,wefirstneedtofindthe
argmaxθ ofthelog-posteriorfunction,i.e.do“standard”deeplearningwithregularizedempirical
MAP
riskminimization.TheonlyadditionalstepistocomputetheinverseoftheHessianmatrixatθ
MAP
(seeFigure1(b)).TheLAcanthereforebeconstructedpost-hoctoapre-trainednetwork,evenone
downloadedoff-the-shelf.Aswediscussbelow,theHessiancomputationcanbeoffloadedtorecently
advancedautomaticdifferentiationlibraries[21].LAsarewidelyusedtoapproximatetheposterior
distributioninlogisticregression[31],Gaussianprocessclassification[32,33],andalsoforBayesian
neuralnetworks(BNNs),bothshallow[34]anddeep[23].Thelatteristhefocusofthiswork.
Generally,anypriorwithtwicedifferentiablelog-densitycanbeused.Duetothepopularityofthe
weight decay regularizer,we assume that the prioris a zero-mean Gaussian p(θ) = N(θ;0,γ2I)
unlessstatedotherwise.2TheHessian∇2L(D;θ)|
|     |     |     |     | θ   |     | θ thendependsbothonthe(simple)log-prior/ |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- |
MAP
regularizerandthe(complicated)log-likelihood/empiricalrisk:
(cid:80)N
|     | ∇2L(D;θ)| |     | =−γ−2I− |     |     | ∇2logp(y | |f  | (x ))| | .   | (7) |
| --- | --------- | --- | ------- | --- | --- | -------- | --- | ------ | --- | --- |
|     |           | θ   | θ MAP   |     |     | n=1 θ    | n   | θ n θ  | MAP |     |
A naive implementation of the Hessian is infeasible because the second term in Eq. (7) scales
quadraticallywiththenumberofnetworkparameters,whichcanbeinthemillionsorevenbillions
[35,36].Inrecentyears,severalworkshaveaddressedscalability,aswellasotherfactorsthataffect
approximationqualityandpredictiveperformanceoftheLA.Inthefollowing,weidentify,review,and
discussfourkeycomponentsthatallowLAstoscaleandperformwellonmoderndeeparchitectures.
SeeFig.2foranoverviewandAppendixBforamoredetailedversionofthereviewanddiscussion.
FourComponentsofScalableLaplaceApproximationsforDeepNeuralNetworks
1 InferenceoverallWeightsorSubsetsofWeights
Inmostcases,itispossibletotreatallweightsprobabilisticallywhenusingappropriateapproxima-
tionsoftheHessian,aswediscussbelowin 2 .AnothersimplewaytoscaletheLAtolargeNNs
2Onecanalsoconsideraper-layerorevenper-parameterweightdecay,whichcorrespondstoamoregeneral,
butstillcomparablysimpleGaussianprior.Inparticular,theHessianofthispriorisstilldiagonalandconstant.
3

|     |     |     |     | 1 WeightstobetreatedprobabilisticallywithLaplace |     |     |     |     |
| --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- |
Deterministicneuralnetworkfθ
|     |     |     |     | (a)All | (b)Subnetwork |     | (c)Last-Layer |     |
| --- | --- | --- | --- | ------ | ------------- | --- | ------------- | --- |
Optional:Trainθasusual(MAP)
|     |     |     | Laplace(.., | subset_of_weights='all'|'subnetwork'|'last_layer') |     |     |     |     |
| --- | --- | --- | ----------- | -------------------------------------------------- | --- | --- | --- | --- |
3 Hyperparametertuningmethod
2 ApproximationoftheHessian
Uanatraaaianaeadafθ
|                  |     |     |             | (a)Full                                           | (b)LRank | (c)KFAC | (d)Diag. |     |
| ---------------- | --- | --- | ----------- | ------------------------------------------------- | -------- | ------- | -------- | --- |
|                  |     |     | Laplace(.., | hessian_structure='full'|'lowrank'|'kron'|'diag') |          |         |          |     |
| (a)OnlineLaplace |     |     | fθ          |                                                   |          |         |          |     |
|                  |     |     | ne d        |                                                   |          |         |          |     |
Trai
4 (Approximate)predictivep(y|fθ(x∗),D)
Classification
Regression
MonteCarlo
|     |     |     |     |     | MonteCarlo | Probitapprox. |     |     |
| --- | --- | --- | --- | --- | ---------- | ------------- | --- | --- |
Exactpredictive
Laplacebridge
(b)Post-hocLaplace
la(x, link_approx='mc'|'probit'|'bridge')
la.optimize_prior_precision()
Figure2:FourkeycomponentstoscaleandapplytheLAtoaneuralnetworkf (withrandomly-
θ
initializedorpre-trainedweightsθ),withcorrespondinglaplacecode. 1 Wefirstchoosewhich
part of the model we want to perform inference over with the LA. We then select how to to
2
approximate the Hessian. 3 We can then perform model selection using the evidence: (a) If we
started with an untrained model f θ ,we can jointly train the model and use the evidence to tune
hyperparametersonline.(b)Ifwestartedwithapre-trainedmodel,wecanusetheevidencetotune
thehyperparameterspost-hoc.Here,shadesrepresentthelosslandscape,whilecontoursrepresent
LAlog-posteriors—fadedcontoursrepresentintermediateiteratesduringhyperparametertuningto
obtainthefinallog-posterior(thickyellowcontours). 4 Finally,tomakepredictionsforanewinput
x ∗ ,wehaveseveraloptionsforcomputing/approximatingthepredictivedistributionp(y|f θ (x ∗ ),D).
(withoutHessianapproximations)isthesubnetworkLA[27],whichonlytreatsasubsetofthemodel
parametersprobabilisticallywiththeLAandleavestheremainingparametersattheirMAP-estimated
values.AnimportantspecialcaseofthisappliestheLAtoonlythelastlinearlayerofanL-layer
NN,whilefixingthefeatureextractordefinedbythefirstL−1layersatitsMAPestimate[37,28].
Thislast-layerLAiscost-effectiveyetcompellingboththeoreticallyandinpractice[28].
HessianApproximationsandTheirFactorizations
2
Oneadvanceinsecond-orderoptimizationthattheLAcanbenefitfromarepositivesemi-definite
approximationstothe(potentiallyindefinite)Hessianofthelog-likelihoodsofNNsinthesecond
termofEq.(7)[38].TheFisherinformationmatrix [39],abbreviatedastheFisheranddefinedby
F := (cid:80)N E [(∇ logp(y|f (x ))| )(∇ logp(y|f (x ))| ) (cid:124) ], (8)
| n=1 | y(cid:98)∼p(y|fθ(xn)) |     | θ   | (cid:98) θ | n θ | θ (cid:98) | θ n θ |     |
| --- | --------------------- | --- | --- | ---------- | --- | ---------- | ----- | --- |
|     |                       |     |     |            | MAP |            | MAP   |     |
isonesuchchoice.3OnecanalsousethegeneralizedGauss-Newtonmatrix(GGN)matrix[41]
|     | (cid:80)N |     | (cid:16)   |     |         | (cid:17) |               |     |
| --- | --------- | --- | ---------- | --- | ------- | -------- | ------------- | --- |
| G:= |           | J(x | ) ∇2logp(y |     | |f)|    | J(x      | ) (cid:124) , | (9) |
|     |           | n=1 | n f        | n   | f=fθMAP | (xn)     | n             |     |
whereJ(x ):=∇ f (x )| istheNN’sJacobianmatrix.AstheFisherandGGNareequivalent
| n θ | θ n | θ   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
MAP
forcommonlog-likelihoods[38],wewillhenceforthrefertotheminterchangeably.IndeepLAs,they
haveemergedasthedefaultchoice[23,24,28,29,27,26,etc.].
3If,insteadoftakingexpectationin(8),weusethetraininglabely
n,wecallthematrixtheempiricalFisher,
whichisdistinctfromtheFisher[38,40].
4

AsF andGarestillquadraticallylarge,wetypicallyneedfurtherfactorizationassumptions. The
most lightweight is a diagonal factorization which ignores off-diagonal elements [42, 43]. More
expressivealternativesareblock-diagonalfactorizationssuchasKronecker-factoredapproximate
curvature (KFAC) [18–20],whichfactorizes eachwithin-layerFisher4 as a Kroneckerproductof
twosmallermatrices.KFAChasbeensuccessfullyappliedtotheLA[23,24]andcanbeimproved
bylow-rankapproximationsoftheKFACfactors[29]byleveragingtheireigendecompositions[44].
Finally,recentworkhasstudied/enabledlow-rankapproximationsoftheHessian/Fisher[45–47].
3 HyperparameterTuning
As with all approximate inference methods, the performance of the LA depends on the (hy-
per)parameters ofthe priorandlikelihood. Forinstance,itis typicallybeneficialto tune the prior
varianceγ2usedforinference[23,28,27,26,22].Commonly,thisisdonethroughcross-validation,
e.g.bymaximizingthevalidationlog-likelihood[23,48]or,additionally,usingout-of-distribution
data [28, 49]. When using the LA,however,marginal likelihood maximization (a.k.a. empirical
Bayesortheevidenceframework[34,50])constitutesamoreprincipledalternativetotunethese
hyperparameters,andrequiresnovalidationdata.Immeretal.[22]showedthatmarginallikelihood
maximizationwithLAcanworkindeeplearningandevenbeperformedinanonlinemannerjointly
withtheMAPestimation.Notethatsuchapproachisnotnecessarilyfeasibleforotherapproximate
inferencemethodsbecausemostdonotprovideanestimateofthemarginallikelihood.Otherrecent
approachesforhyperparametertuningfortheLAincludeBayesianoptimization[51]ortheaddition
ofdedicated,trainablehiddenunitsforthesolepurposeofuncertaintytuning[49].
4 ApproximatePredictiveDistribution
To predict using a posterior (approximation) p(θ|D), we need to compute p(y|f(x ),D) =
∗
(cid:82) p(y|f (x ))p(θ|D)dθ for any test point x ∈ Rn, which is intractable in general. The sim-
θ ∗ ∗
plest but most general approximation to p(y|x ,D) is Monte Carlo integration using S samples
∗
(θ )S from p(θ|D): p(y|f(x ),D) ≈ S−1(cid:80)S p(y|f (x )). However,forLAs with GGN
s s=1 ∗ s=1 θs ∗
andFisherHessianapproximationsMonteCarlointegrationcanperformpoorly[48,26].Immeretal.
[26]attributethistotheinconsistencybetweenHessianapproximationandthepredictiveandsuggest
tousealinearizedpredictiveinstead,whichcanalsobeusefulfortheoreticanalyses[28].Forthe
last-layerLA,theHessiancoincideswiththeGGNandthelinearizedpredictiveisexact.
ThepredictiveofalinearizedneuralnetworkwithaLAapproximationtotheposteriorp(θ|D)≈
N(θ;θ ,Σ)resultsinaGaussiandistributiononneuralnetworkoutputsf :=f(x )andtherefore
MAP ∗ ∗
enables simple approximations or even a closed-form solution. The distribution on the outputs
is given byp(f |x ,D) ≈ N(f ;f (x ),J(x ) (cid:124) ΣJ(x )) andis typicallysignificantlylower-
∗ ∗ ∗ θ ∗ ∗ ∗
MAP
dimensional(numberofoutputsCinsteadofparametersD).Itcanalsobeinferredentirelyinfunction
spaceasaGaussianprocess[25,26].Giventhedistributiononoutputsf ,thepredictivedistribution
∗
(cid:82)
canbeobtainedbyintegrationagainstthelikelihood:p(y|x ,D) = p(y|f )p(f |x ,D)dθ.In
∗ ∗ ∗ ∗
thecaseofregressionwithaGaussianlikelihoodwithvarianceσ2,thesolutioncanevenbeobtained
analytically:p(y|x ,D)≈N(y;f (x ),J(x ) (cid:124) ΣJ(x )+σ2I).Fornon-Gaussianlikelihoods,
∗ θ ∗ ∗ ∗
MAP
e.g.inclassification,afurtherapproximationisneeded.Again,thesimplestapproximationtothis
isMonteCarlointegration. Inthebinarycase,wecanemploytheprobitapproximation[31,16]
whichapproximatesthelogisticfunction withtheprobitfunction. In themulti-classcase,wecan
useitsgeneralization,theextendedprobitapproximation[52].Finally,firstproposedfornon-BNN
applications[53,54],theLaplacebridgeapproximatesthesoftmax-GaussianintegralviaaDirichlet
distribution[55].Thekeyadvantageisthatityieldsadistributionoftheintegralsolutions.
3 laplace:AToolkitforDeepLaplaceApproximations
ImplementingtheLAisnon-trivial,asitrequiresefficientcomputationandstorageoftheHessian.
Whilethisisnotfundamentallydifficult,thereexistsnocomplete,easy-to-use,andstandardizedim-
plementationofvariousLAflavors—instead,itiscommonfordeeplearningresearcherstorepeatedly
re-implementthe LA andHessian computation withvarying efficiency[56–58,etc.]. An efficient
implementation typically requires hundreds of lines of code,making it hard to quickly prototype
4TheelementsF orGcorrespondingtotheweightW
l
⊆θofthel-thlayerofthenetwork.
5

| from laplace         | import Laplace   |     |     | from laplace | import Laplace       |     |
| -------------------- | ---------------- | --- | --- | ------------ | -------------------- | --- |
| 1                    |                  |     |     | 1            |                      |     |
| 2                    |                  |     |     | 2            |                      |     |
| 3 # Load pre-trained | model            |     |     | 3 # Load un- | or pre-trained model |     |
| 4 model =            | load_map_model() |     |     | 4 model =    | load_map_model()     |     |
| 5                    |                  |     |     | 5            |                      |     |
# Define and fit LA variant with custom settings # Fit default, recommended LA variant:
| 6                     |                           |     |     | 6                      |               |     |
| --------------------- | ------------------------- | --- | --- | ---------------------- | ------------- | --- |
| 7 la = Laplace(model, | 'classification',         |     |     | 7 # Last-layer         | KFAC LA       |     |
|                       | subset_of_weights='all',  |     |     | la = Laplace(model,    | 'regression') |     |
| 8                     |                           |     |     | 8                      |               |     |
| 9                     | hessian_structure='diag') |     |     | 9 la.fit(train_loader) |               |     |
la.fit(train_loader)
| 10  |     |     |     | 10  |     |     |
| --- | --- | --- | --- | --- | --- | --- |
11 la.optimize_prior_precision(method='CV', 11 # Differentiate marginal likelihood w.r.t.
12 val_loader=val_loader) 12 # prior precision and observation noise
ml = la.marglik(prior_precision=prior_prec,
| 13  |     |     |     | 13  |     |     |
| --- | --- | --- | --- | --- | --- | --- |
14 # Make prediction with custom predictive approx. 14 sigma_noise=obs_noise)
| pred = la(x, | pred_type='glm', | link_approx='probit') |     | ml.backward() |     |     |
| ------------ | ---------------- | --------------------- | --- | ------------- | --- | --- |
| 15           |                  |                       |     | 15            |     |     |
Listing1:FitdiagonalLAoverallweightsof Listing2:FitKFACLAoverthelastlayerofa
apre-trainedclassificationmodel,dopost-hoc pre-orun-trainedregressionmodelanddifferen-
tuningofthepriorprecisionhyperparameterus- tiateitsmarginallikelihoodw.r.t.somehyperpa-
ingcross-validation,andmakeapredictionfor rametersforpost-hochyperparametertuningor
onlineempiricalBayes(seeImmeretal.[22]).
inputxwiththeprobitapproximation.
withthe LA. To address this,we introduce laplace: a simple,easy-to-use,extensible library for
scalableLAsofdeepNNsinPyTorch[59].laplaceenablesallsensiblecombinationsofthefour
componentsdiscussedinSection2—seeFig.2fordetails.Listings1and2showcodeexamples.
ThecoreoflaplaceconsistsofefficientimplementationsoftheLA’skeyquantities:(i)posterior
(i.e.Hessiancomputationandstorage),(ii)marginallikelihood,and(iii)posteriorpredictive.For(i),
totakeadvantageofadvancesinautomaticdifferentiation,weoutsourcetheHessiancomputation
tostate-of-the-art,optimizedsecond-orderoptimizationlibraries:BackPACK[21]andASDL[60].
Moreover,we design laplace in a modularmannerthat makes it easy to addnew backends and
approximationsinthefuture.For(ii),wefollowImmeretal.[22]inourimplementationoftheLA’s
marginal likelihood—it is thus both efficient and differentiable and allows the user to implement
bothonlineandpost-hocmarginallikelihoodtuning,cf.Listing2.Notethatlaplacealsosupports
standardcross-validationforhyperparametertuning[23,28],asshowninListing1.Finally,for(iii),
laplacesupportsallapproximationstotheposteriorpredictivedistributiondiscussedinSection2—it
thusprovidestheuserwithflexibilityinmakingpredictions,dependingonthecomputationalbudget.
| Default | behavior |               |              |        |                               |        |
| ------- | -------- | ------------- | ------------ | ------ | ----------------------------- | ------ |
|         | To       | abstract away | from a large | number | of options available (Section | 2), we |
providethefollowingdefaultchoicesbasedonourextensiveexperiments(Section4);theyshouldbe
applicableandperformdecentlyinthemajorityofusecases:weassumeapre-trainednetworkand
treatonlythelast-layerweightsprobabilistically(last-layerLA),usetheKFACfactorizationofthe
GGNandtunethehyperparameterspost-hocusingempiricalBayes.Tomakepredictions,weusethe
closed-formGaussianpredictivedistributionforregressionandthe(extended)probitapproximation
forclassification.Ofcourse,theusercanpickcustomchoices(Listings1and2).
Limitations Because laplace employs external libraries (BackPACK [21] and ASDL [60]) as
backends,itinheritstheavailablechoicesofHessianfactorizationsfromtheselibraries.Forinstance,
theLAvariantproposedbyLeeetal.[29]cancurrentlynotbeimplementedvialaplace,because
neitherbackendsupportseigenvalue-correctedKFAC[44](yet).
4 Experiments
WebenchmarkvariousLAsimplementedvialaplace.Section4.1addressesthequestionof“which
are the best design choices for the LA”, in light of Figure 2. Section 4.2 shows that the LA is
competitive to strong Bayesian baselines in in-distribution, dataset-shift, and out-of-distribution
(OOD)settings.WethenshowcasesomeapplicationsoftheLAindownstreamtasks.Section4.3
demonstratestheapplicabilityofthe(last-layer)LAonvariousdatamodalitiesandNNarchitectures
(includingtransformers[61])—settingswhereotherBayesianmethodsarechallengingtoimplement.
Section4.4showshowtheLAcanbeusedasaneasy-to-useyetstrongbaselineincontinuallearning.
Inallresults,arrowsbehindmetricnamesdenoteiflower(↓)orhigher(↑)valuesarebetter.
6

94
92
90
91 92 93
Acc.(ID)↑
↑)DOO(CORUA
CIFAR-10+DA CIFAR-10
90 Table 1: OOD detection performance aver-
aged over all test sets (see Appendix C.2 for
85
details). Confidence is defined as the max.
80 online of the predictive probability vector [62] (e.g.
post-hoc
Confidence([0.7,0.2,0.1]) = 0.7). LA and MAP
75
82 86 90 especially LA* reduce the overconfidence of
MAP and achieve better results than the VB,
Acc.(ID)↑
Figure3:In-vs.out-of-distribution(IDandOOD, CSGHMC(HMC),andSWAG(SWG)baselines.
resp.)performanceonCIFAR-10ofdifferentLA
Confidence↓ AUROC↑
configurations (dots), each being a combination
Methods MNIST CIFAR-10 MNIST CIFAR-10
ofsettingsfor1)subset-of-weights,2)covariance
structure,3)hyperparametertuning,and4)predic- MAP 75.0±0.4 76.1±1.2 96.5±0.1 92.1±0.5
DE 65.7±0.3 65.4±0.4 97.5±0.0 94.0±0.1
tiveapproximation(seeAppendixC.1fordetails).
VB 73.2±0.8 58.8±0.7 95.8±0.2 88.7±0.3
“DA”standsfor“dataaugmentation”.Post-hocper- HMC 69.2±1.7 69.4±0.6 96.1±0.2 90.6±0.2
formsbetterwithDAandastrongpre-trainednet- SWG 75.8±0.3 68.1±2.3 96.5±0.1 91.3±0.8
work, while online performs better without DA LA 67.5±0.4 69.0±1.3 96.2±0.2 92.2±0.5
whereoptimalhyperparametersareunknown. LA* 56.1±0.5 55.7±1.2 96.4±0.2 92.4±0.5
4.1 ChoosingtheRightLaplaceApproximation
InSection2wepresentedmultipleoptionsforeachcomponentofthedesignspaceoftheLA,resulting
inalargenumberofpossiblecombinations,allofwhicharesupportedbylaplace. Here,wetry
to reduce this complexity and make suggestions for sensible default choices that cover common
applicationscenarios.Tothisend,weperformedacomprehensivecomparisonbetweenmostvariants;
wemeasuredin-andout-of-distributionperformanceonstandardimageclassificationbenchmarks
(MNIST, FashionMNIST, CIFAR-10) but also considered the computational complexity of each
variant.WeprovidedetailsofthecomparisonandalistoftheconsideredvariantsinAppendixC.1
andsummarizethemainargumentsandtake-awaysinthefollowing.
Hyperparametertuningandparameterinference. WecanapplytheLApurelypost-hoc(only
tunehyperparametersofapre-trainednetwork)oronline(tunehyperparametersandtrainthenetwork
jointly, as e.g. suggested by Immer et al. [22]). We find that the online LA only works reliably
whenitisappliedtoallweightsofthenetwork.Incontrast,applyingtheLApost-hoconlyonthe
lastlayerinsteadofallweightstypicallyyieldsbetterperformanceduetolessunderfitting,andis
significantly cheaper. For problems where a pre-trained network or optimal hyperparameters are
available,e.g. for well-studied data sets,we,therefore,suggest using the post-hoc variant on the
last layer. This LA has the benefit that it has minimal overhead over a standard neural network
forwardpass(cf.Fig.5)whileperformingonparorbetterthanstate-of-the-artapproaches(cf.Fig.4).
Whenhyperparametersareunknownornovalidationdataisavailable,wesuggesttrainingtheneural
networkonlinebyoptimizingthemarginallikelihood,followingImmeretal.[22](cfSection4.4).
Figure 3 illustrates this on CIFAR-10: for CIFAR-10 with data augmentation,strong pre-trained
networksandhyperparametersareavailableandthepost-hocmethodsdirectlyprofitfromthatwhile
theonlinemethodsmerelyreachthesameperformance.OnthelessstudiedCIFAR-10withoutdata
augmentation,theonlinemethodcanimprovetheperformanceoverthepost-hocmethods.
Covarianceapproximationandstructure. Generally,wefindthatamoreexpressivecovariance
approximationimprovesperformance,aswouldbeexpected.However,afullcovarianceisinmost
casesintractableforfullnetworksornetworkswithlargelastlayers.TheKFACstructuredcovariance
provides a good trade-off between expressiveness and speed. Diagonal approximations perform
significantlyworsethanKFACandarethereforenotsuggested.Independentofthestructure,wefind
thattheempiricalFisher(EF)approximationsperformbetteronout-of-distributiondetectiontasks
whileGGNapproximationstendtoperformbetteronin-distributionmetrics.
Predictive distribution. Considering in- and out-of-distribution (OOD) performance as well as
cost,theprobitprovidesthebestapproximationtothepredictiveforthelast-layerLA.MCintegration
cansometimesbesuperiorforOODdetectionbutatanincreasedcomputationalcost.TheLaplace
bridgehasthesamecostastheprobitapproximation buttypicallyprovidesinferiorresultsin our
7

| 100 |     |     | 100% |     |     |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- | --- | --- |
0.6
| MAP  | DE VB  | HMC |     | 7.5 |     |     |     |     |
| ---- | ------ | --- | --- | --- | --- | --- | --- | --- |
| SWG  | LA LA* |     |     |     |     | 0.4 |     |     |
| 10−1 |        |     | 99% | 5   |     |     |     |     |
0.2
2.5
| 10−2 |     |     |     | 0   |     | 0   |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- |
98%
| NLL↓ | ECE↓ | %Acc.↑ |     | 0 50 | 100 150 | 0   | 50  | 100 150 |
| ---- | ---- | ------ | --- | ---- | ------- | --- | --- | ------- |
| 100  |      |        | 96% |      |         |     |     |         |
|      |      |        |     | 2    |         | 0.3 |     |         |
| 10−1 |      |        | 93% |      |         | 0.2 |     |         |
1
0.1
10−2
|      |         |        | 90% | 0              |       | 0   |                |         |
| ---- | ------- | ------ | --- | -------------- | ----- | --- | -------------- | ------- |
| NLL↓ | ECE↓    | %Acc.↑ |     | 0 1            | 2 3 4 | 5 0 | 1              | 2 3 4 5 |
|      | Metrics |        |     | ShiftIntensity |       |     | ShiftIntensity |         |
(a)In-Distribution (b)Distribution-shiftNLL↓ (c)Distribution-shiftECE↓
Figure4:Assessingmodelcalibration(a)onin-distributiondataand(b,c)underdistributionshift,for
theMNIST(toprow)andCIFAR-10(bottomrow)datasets.For(b,c),weusetheRotated-MNIST(top)
andCorrupted-CIFAR-10(bottom)benchmarks[63,64].In(a),wereportaccuracyand,tomeasure
calibration,negativelog-likelihood(NLL)andexpectedcalibrationerror(ECE)—allevaluatedon
thestandardtestsets.In(b)and(c),weplotshiftintensitiesagainstNLLandECE,respectively.For
Rotated-MNIST(top),shiftintensitiesdenotedegreesofrotationoftheimages,whileforCorrupted-
(a)
CIFAR-10 (bottom),theydenote the amountofimage distortion (see [63, 64] fordetails). On
in-distribution data, LA is the best-calibrated method in terms of ECE, while also retaining the
accuracyofMAP(unlikeVBandCSGHMC).(b,c)Oncorrupteddata,allBayesianmethodsimprove
uponMAPsignificantly.Eventhoughpost-hoc,allLAsachievecompetitiveresults,eventoDE.In
particular,LA*achievesthebestresults,attheexpenseofslightlyworsein-distributioncalibration—
thistrade-offbetweenin-andout-of-distributionperformancehasbeenobservedpreviously[65].
experiments.WhenusingtheLAonlinetooptimizehyperparameters,wefindthattheresultingMAP
predictiveprovidesgoodperformancein-distribution,butaprobitorMCpredictiveimprovesOOD
performance.
Overallrecommendation. Followingtheexperimentalevidence,thedefaultinlaplaceisapost-
hocKFAClast-layerLAwithaGGNapproximationtotheHessian.Thisdefaultisapplicabletoall
architecturesthathaveafully-connectedlastlayerandcanbeeasilyappliedtopre-trainednetworks.
Forproblemswheretrainednetworksareunavailableorhyperparametersareunknown,theonline
KFACLAwithaGGNorempiricalFisherprovidesagoodbaselinewithminimaleffort.
4.2 PredictiveUncertaintyQuantification
We consider two flavors of LAs: the default flavor of laplace (LA) and the most robust one in
termsofdistributionshiftfoundinSection4.1(LA*—last-layer,withafullempiricalFisherHessian
approximation,andtheprobitapproximation).WecomparethemwiththeMAPnetwork(MAP)and
variouspopularandstrongBayesianbaselines:DeepEnsemble[DE,14],mean-fieldvariationalBayes
[VB,11,12]withtheflipoutestimator[66],cyclicalstochastic-gradientHamiltonianMonteCarlo
[CSGHMC / HMC,67],and SWAG [SWG,15]. Foreach baseline,we use the hyperparameters
recommendedintheoriginalpaper—seeAppendixAfordetails.First,Fig.4showsthatLAandLA*
are,respectively,competitivewithandsuperiortothebaselinesintrading-offbetweenin-distribution
calibrationanddataset-shiftrobustness.Second,Table1showsthatLAandLA*achievebetterresults
onout-of-distribution(OOD)detectionthanevenVB,CSGHMC,andSWG.
TheLAshinesevenmorewhenweconsiderits(timeandmemory)costrelativetotheother,more
complexbaselines. In Fig. 5 we showthe wall-clocktimes ofeachmethodrelative to MAP’s for
training and prediction. As expected, DE, VB, and CSGHMC are slow to train and in making
predictions:theyarebetweentwotofivetimesmoreexpensivethanMAP.Meanwhile,despitebeing
post-hoc,SWGisalmosttwiceasexpensiveasMAPduringtrainingduetotheneedforsampling
8

|     |        |     | MAP | DE  | Temp.Scaling | LA     |        |
| --- | ------ | --- | --- | --- | ------------ | ------ | ------ |
|     | ID OOD | ID  | OOD |     | ID OOD       | ID OOD | ID OOD |
0.3
0.70
| 0.75 |     | 3   |     |     |     | 0.70 |     |
| ---- | --- | --- | --- | --- | --- | ---- | --- |
↓LLN
| 0.50 |     |     |     | 0.2 |     |     | 0.65 |
| ---- | --- | --- | --- | --- | --- | --- | ---- |
0.65
2
0.25
|                 |     |     |     | 0.1 |     | 0.60 | 0.60 |
| --------------- | --- | --- | --- | --- | --- | ---- | ---- |
| ↓.bilaC/ECE 0.2 |     | 0.4 |     |     |     |      | 40   |
0.3
0.10
|     |     | 0.2 |     | 0.2 |     |      |     |
| --- | --- | --- | --- | --- | --- | ---- | --- |
| 0.1 |     |     |     |     |     |      | 20  |
|     |     |     |     | 0.1 |     | 0.05 |     |
0.0
(a) Camelyon17 (b) FMoW (c) CivilComments (d) Amazon (e) PovertyMap
Figure6:Assessingreal-worlddistributionshiftrobustnessonfivedatasetsfromtheWILDSbench-
mark[68],coveringdifferentdatamodalities,modelarchitectures,andoutputtypes.Camelyon17:
Tissueslideimagetumorclassificationacrosshospitals(DenseNet-121[69]).FMoW:Satelliteimage
land use classification across regions/years (DenseNet-121). CivilCommments: Online comment
toxicityclassificationacrossdemographics(DistilBERT[70]). Amazon:Productreviewsentiment
classificationacrossusers(DistilBERT).PovertyMap:Satelliteimageassetwealthregressionacross
countries(ResNet-18[35]).Weplotmeans±standarderrorsoftheNLL(top)andECE(forclassifi-
cation)orregressioncalibrationerror[71](bottom).Thein-distribution(leftpanels)andOOD(right
panels)datasetsplitscorrespondtodifferentdomains(e.g.hospitalsforCamelyon17).LAismuch
bettercalibratedthanMAP,andcompetitivewithtemp.scalingandDE,especiallyontheOODsplits.
andupdatingitsbatchnormalizationstatistics.Moreover,with30samples,asrecommendedbyits
authors[15],itisveryexpensiveatpredictiontime—morethantentimesmoreexpensivethanMAP.
Meanwhile,LA(andLA*)isthecheapestofallmethods
| considered:itonlyincursanegligibleoverheadontopof |     |     |     |     |     | ↓emiTevitaleR Training |     |
| ------------------------------------------------- | --- | --- | --- | --- | --- | ---------------------- | --- |
10
| thecostsofMAP.Thisissimilarforthememoryconsump-    |     |     |     |     |     | Prediction |               |
| -------------------------------------------------- | --- | --- | --- | --- | --- | ---------- | ------------- |
| tion(seeTable5inAppendixC.5).ThisshowsthattheLA    |     |     |     |     |     | 5          |               |
| issignificantlymorememory-andcompute-efficientthan |     |     |     |     |     | 2          |               |
| alltheothermethods,addingminimaloverheadoverMAP    |     |     |     |     |     | 0          |               |
|                                                    |     |     |     |     |     | MAP DE     | VB HMC SWG LA |
inferenceandprediction.ThismakestheLAparticularly
attractiveforpractitioners,especiallyinlow-resourceen- Figure5:Wall-clocktimecostsrelative
vironments.TogetherwithFig.4andTable1,thisjustifies toMAP.LAintroducesnegligibleover-
ourdefaultflavorinlaplace,andimportantly,showsthat headoverMAP,whileallotherbaselines
Bayesiandeeplearningdoesnothavetobeexpensive. aresignificantlymoreexpensive.
4.3 RealisticDistributionShift
Sofar,ourexperimentsfocusedoncomparablysimplebenchmarks,allowingustocomprehensively
assessdifferentLAvariantsandcomparetomoreinvolvedBayesianmethodssuchasVB,MCMC,
andSWAG.Inmorerealisticsettings,however,wherewewanttoimprovetheuncertaintyofcomplex
andcostly-to-trainmodels,suchastransformers[61],thesemethodswouldlikelybedifficulttoget
to workwellandexpensive to run. However,one mightoften have access to a pre-trainedmodel,
allowingforthecheapuseofpost-hocmethodssuchastheLA.Todemonstratethis,weshowhow
laplacecanimprovethedistributionshiftrobustnessofcomplexpre-trainedmodelsinlarge-scale
settings. To this end,we use WILDS [68],a recently proposed benchmark of realistic distribution
shiftsencompassingavarietyofreal-worlddatasetsacrossdifferentdatamodalitiesandapplication
domains.WhiletheWILDSmodelsemploycomplex(e.g.convolutionalortransformer)architectures
asfeatureextractors,theyallfeedintoalinearoutputlayer,allowingustoconvenientlyandcheaply
applythelast-layerLA.Asbaselines,weconsider:1)thepre-trainedMAPmodels[68],2)post-hoc
temperaturescalingoftheMAPmodels(forclassificationtasks)[1],and3)deepensembles[14].5
5Wesimplyconstructdeepensemblesfromthevariouspre-trainedmodelsprovidedbyKohetal.[68].
9

MoredetailsontheexperimentalsetupareprovidedinAppendixC.3.Fig.6showstheresultsonfive
differentWILDSdatasets(seecaptionfordetails).Overall,Laplaceissignificantlybettercalibrated
thanMAP,andcompetitivewithtemperaturescalingandensembles,especiallyontheOODsplits.
4.4 FurtherApplications
1
0.9
0.8
0.7
2 4 6 8 10
Task
↑ycaruccA
Beyond predictive uncertainty quantification, the LA is
useful in wide range of applications such as Bayesian
optimization [37],bandits [72],active learning [34, 73],
and continual learning [24]. The laplace library con-
MAP VB(VOGN)
veniently facilitates these applications. As an example, LA-Diag LA-KFAC
we demonstrate the performance of the LA on the stan-
dard continual learning benchmark with the Permuted-
MNIST dataset,consisting of ten tasks each containing
Figure7:Continuallearningresultson
pixel-permutedMNISTimages[74].Figure7showshow
Permuted-MNIST.MAPfailscatastroph-
the all-layer diagonal and Kronecker-factored LAs can
ically as more tasks are added. The
overcomecatastrophicforgetting.Inthisexperiment,we
Bayesian approaches substantially out-
updatetheLAsaftereachtaskassuggestedbyRitteretal.
performMAP,withLA-KFACperform-
[24]andimproveupontheirresultbytuningthepriorpre-
ingthebest,closelyfollowedbyVOGN.
cision through marginal likelihood optimization during
training,followingImmeretal.[22](detailsinAppendixC.4).Usingthisscheme,theperformance
after10 tasks is at around 96% accuracy,outperforming otherBayesian approaches forcontinual
learning[7,75,76].Concretely,weshowthattheKFACLA,whilemuchsimplerwhenappliedvia
laplace,can achieve better performance to a recent VB baseline [VOGN,13]. Our library thus
providesaneasyandquickwayofconstructingastrongbaselineforthisapplication.
5 RelatedWork
TheLAisfundamentallyalocalapproximationthatcoversasinglemodeoftheposterior;similarly,
otherGaussianapproximationssuchasmean-fieldvariationalinference[11–13]orSWAG[15]also
onlycapturelocalinformation.SWAGusesthefirstandsecondempiricalmomentofSGDiterates
toformadiagonalpluslow-rankGaussianapproximationbutrequiresstoringmanyNNcopiesand
applyinga(costly)heuristicrelatedtobatchnormalizationattesttime.Incontrast,theLAdirectlyuses
curvatureinformationofthelossaroundtheMAPandcanbeappliedpost-hoctopre-trainedNNs.
In contrast to local Gaussian approximations, (stochastic-gradient) MCMC methods [77, 78, 67,
79,80,etc.]anddeepensembles[14]canexploreseveralmodes.Nevertheless,priorworks—also
validatedinourexperimentsinSection4—indicatethatusingasinglemodemightnotbeaslimiting
inpracticeasonemightthink.WilsonandIzmailov[81]conjecturethatthisisduetothecomplex,
nonlinearconnectionbetweentheparameterspaceandthefunction(output)spaceofNNs.Moreover,
whileunbiasedcomparedtoitssimpleralternatives,MCMCmethodsarenotoriouslyexpensivein
practiceand,thus,oftenrequirefurtherapproximationssuchasdistillation[82,83].Finally,notethat
boththeLAaswellasSWAGcanbeextendedtoensemblesofmodesinapost-hocmanner[84,81].
6 Conclusion
In this paper,we arguedthatthe Laplace approximation is a simple yetcompetitive andversatile
methodforBayesiandeeplearningthatdeserveswideradoption.Tothisend,wereviewedmanyrecent
advancestoandvariantsoftheLaplaceapproximation,includingversionswithminimalcostoverhead
thatcanbeappliedpost-hoctopre-trainedoff-the-shelfmodels.Inacomprehensiveevaluationwe
demonstrated that the Laplace approximation is on par with other approaches that approximate
the intractable network posterior, but at typically much lower computational cost. A particularly
simple variant that only treats some weights probabilistically can even be used in the context of
pre-trainedtransformermodelstoimprovepredictiveuncertainty.Asanefficientimplementationis
notstraightforward,weintroducedlaplace,amodularandextensiblesoftwarelibraryforPyTorch
offeringuser-friendlyaccesstoallmajorflavorsoftheLaplaceapproximation.Inthisway,Laplace
approximationsprovidedrop-inBayesianfunctionalityformosttypesofdeepneuralnetworks.
10

AcknowledgmentsandDisclosureofFunding
We thank Kazuki Osawa for providing early access to his automatic second-order differentiation
(ASDL)libraryforPyTorch,AlexBotevforfeedbackonourmanuscript,andSimoRyuforspotting
asigntypoinEq.(5).Wealsothanktheanonymousreviewersfortheirhelpfulsuggestions.
E.D.acknowledgesfundingfromtheEPSRCandQualcomm.A.I.gratefullyacknowledgesfundingby
theMaxPlanckETHCenterforLearningSystems(CLS).R.E.,A.K.andP.H.gratefullyacknowledge
financialsupportbytheEuropeanResearchCouncilthroughERCStGAction757275/PANAMA;
theDFGClusterofExcellence“MachineLearning-NewPerspectivesforScience”,EXC2064/1,
projectnumber390727645;theGermanFederalMinistryofEducationandResearch(BMBF)through
theTübingenAICenter(FKZ:01IS18039A);andfundsfromtheMinistryofScience,Researchand
ArtsoftheStateofBaden-Württemberg.A.K.isgratefultotheInternationalMaxPlanckResearch
SchoolforIntelligentSystems(IMPRS-IS)forsupport.
References
[1] ChuanGuo,GeoffPleiss,YuSun,andKilianQ.Weinberger. OnCalibrationofModernNeuralNetworks.
InICML,2017.
[2] JamesKirkpatrick,RazvanPascanu,NeilRabinowitz,JoelVeness,GuillaumeDesjardins,AndreiARusu,
KieranMilan,JohnQuan,TiagoRamalho,AgnieszkaGrabska-Barwinska,etal. OvercomingCatastrophic
ForgettinginNeuralNetworks. ProceedingsoftheNationalAcademyofSciences,114(13),2017.
[3] FrankHutter,LarsKotthoff,andJoaquinVanschoren. AutomatedCachineLearning:Methods,Systems,
Challenges. SpringerNature,2019.
[4] DavidBarber. BayesianReasoningandMachineLearning. CambridgeUniversityPress,2012.
[5] ZoubinGhahramani. ProbabilisticMachineLearningandArtificialIntelligence. Nature,521(7553),2015.
[6] YarinGalandZoubinGhahramani.DropoutasaBayesianApproximation:RepresentingModelUncertainty
inDeepLearning. InICML,2016.
[7] CuongVNguyen,YingzhenLi,ThangDBui,andRichardETurner. VariationalContinualLearning. In
ICLR,2018.
[8] DavidJCMacKay.ProbableNetworksandPlausiblePredictions—aReviewofPracticalBayesianMethods
forSupervisedNeuralNetworks. Network:ComputationinNeuralSystems,1995.
[9] YarinGal. Uncertaintyindeeplearning. UniversityofCambridge,2016.
[10] Geoffrey E Hinton and Drew Van Camp. Keeping the Neural Networks Simple by Minimizing the
DescriptionLengthoftheWeights. InCOLT,1993.
[11] AlexGraves. PracticalVariationalInferenceforNeuralNetworks. InNIPS,2011.
[12] CharlesBlundell,JulienCornebise,KorayKavukcuoglu,andDaanWierstra. WeightUncertaintyinNeural
Networks. InICML,2015.
[13] Kazuki Osawa,Siddharth Swaroop,Mohammad Emtiyaz E Khan,Anirudh Jain,Runa Eschenhagen,
Richard E Turner,and Rio Yokota. Practical Deep Learning with Bayesian Principles. In NeurIPS,
2019.
[14] BalajiLakshminarayanan,AlexanderPritzel,andCharlesBlundell. SimpleandScalablePredictiveUncer-
taintyEstimationusingDeepEnsembles. InNIPS,2017.
[15] WesleyJMaddox,PavelIzmailov,TimurGaripov,DmitryPVetrov,andAndrewGordonWilson.ASimple
BaselineforBayesianUncertaintyinDeepLearning. InNeurIPS,2019.
[16] DavidJCMacKay. BayesianInterpolation. Neuralcomputation,4(3),1992.
[17] Pierre-SimonLaplace. MémoiresdeMathématiqueetdePhysique,TomeSixieme. 1774.
[18] TomHeskes. On“Natural”LearningandPruninginMultilayeredPerceptrons. NeuralComputation,12
(4),2000.
11

[19] JamesMartensandRogerGrosse. OptimizingNeuralNetworkswithKronecker-FactoredApproximate
Curvature. InICML,2015.
[20] AleksandarBotev,HippolytRitter,andDavidBarber. PracticalGauss-NewtonOptimisationforDeep
Learning. InICML,2017.
[21] FelixDangel,FrederikKunstner,andPhilippHennig. Backpack:PackingMoreintoBackprop. InICLR,
2020.
[22] AlexanderImmer,MatthiasBauer,VincentFortuin,GunnarRätsch,andMohammadEmtiyazKhan. Scal-
ableMarginalLikelihoodEstimationforModelSelectioninDeepLearning. InICML,2021.
[23] HippolytRitter,AleksandarBotev,andDavidBarber. A Scalable Laplace Approximation forNeural
Networks. InICLR,2018.
[24] HippolytRitter,AleksandarBotev,andDavidBarber. Online StructuredLaplace Approximations for
OvercomingCatastrophicForgetting. InNIPS,2018.
[25] MohammadEmtiyazEKhan,AlexanderImmer,EhsanAbedi,andMaciejKorzepa.ApproximateInference
TurnsDeepNetworksIntoGaussianProcesses. InNeurIPS,2019.
[26] AlexanderImmer,MaciejKorzepa,andMatthiasBauer. ImprovingPredictionsofBayesianNeuralNet-
worksviaLocalLinearization. InAISTATS,2021.
[27] ErikDaxberger,EricNalisnick,JamesUrquhartAllingham,JavierAntorán,andJoséMiguelHernández-
Lobato. BayesianDeepLearningviaSubnetworkInference. InICML,2021.
[28] AgustinusKristiadi,MatthiasHein,andPhilippHennig. BeingBayesian,EvenJustaBit,FixesOverconfi-
denceinReLUNetworks. InICML,2020.
[29] JongseokLee,MatthiasHumt,JianxiangFeng,andRudolphTriebel. EstimatingModelUncertaintyof
NeuralNetworksinSparseInformationForm. InICML,2020.
[30] PavelIzmailov,WesleyJMaddox,PolinaKirichenko,TimurGaripov,DmitryVetrov,andAndrewGordon
Wilson. SubspaceInferenceforBayesianDeepLearning. InUAI,2019.
[31] DavidJSpiegelhalterandSteffenLLauritzen.SequentialUpdatingofConditionalProbabilitiesonDirected
GraphicalStructures. Networks,1990.
[32] Christopher KI Williams andDavidBarber. Bayesian Classification withGaussian processes. IEEE
TransactionsonPatternAnalysisandMachineIntelligence,20(12),1998.
[33] CarlEdwardRasmussenandChristopherK.I.Williams. GaussianProcessesinMachineLearning. The
MITPress,2005.
[34] DavidJCMacKay. TheEvidenceFrameworkAppliedtoClassificationNetworks. Neuralcomputation,
1992.
[35] KaimingHe,XiangyuZhang,ShaoqingRen,andJianSun.DeepResidualLearningforImageRecognition.
InCVPR,2016.
[36] MohammadShoeybi,MostofaPatwary,RaulPuri,PatrickLeGresley,JaredCasper,andBryanCatanzaro.
Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv
preprintarXiv:1909.08053,2019.
[37] JasperSnoek,OrenRippel,KevinSwersky,RyanKiros,NadathurSatish,NarayananSundaram,Mostofa
Patwary,MrPrabhat,andRyanAdams. Scalablebayesianoptimizationusingdeepneuralnetworks. In
ICML,2015.
[38] James Martens. New insights and perspectives on the natural gradient method. Journal of Machine
LearningResearch,21(146):1–76,2020.
[39] Shun-IchiAmari. NaturalGradientWorksEfficientlyinLearning. Neuralcomputation,10(2),1998.
[40] FrederikKunstner,LukasBalles,andPhilippHennig. LimitationsoftheEmpiricalFisherApproximation
forNaturalGradientDescent. InNeurIPS,2019.
[41] NicolNSchraudolph. FastCurvatureMatrix-VectorProductsforSecond-OrderGradientDescent. Neural
computation,14(7),2002.
12

| [42] YannLeCun,JohnSDenker,andSaraASolla. |     | OptimalBrainDamage. |     | InNIPS,1990. |
| ----------------------------------------- | --- | ------------------- | --- | ------------ |
[43] JohnSDenkerandYannLeCun. TransformingNeural-NetOutputLevelstoProbabilityDistributions. In
NIPS,1990.
[44] ThomasGeorge,CésarLaurent,XavierBouthillier,NicolasBallas,andPascalVincent. FastApproximate
| NaturalGradientDescentinaKroneckerFactoredEigenbasis. |     |     | InNIPS,2018. |     |
| ----------------------------------------------------- | --- | --- | ------------ | --- |
[45] DavidMadras,JamesAtwood,andAlexD’Amour. Detectingextrapolationwithlocalensembles. InICLR,
2020.
[46] WesleyJMaddox,GregoryBenton,andAndrewGordonWilson. Rethinkingparametercountingindeep
| models:Effectivedimensionalityrevisited. |     | arXivpreprintarXiv:2003.02139,2020. |     |     |
| ---------------------------------------- | --- | ----------------------------------- | --- | --- |
[47] ApoorvaSharma,NavidAzizan,andMarcoPavone. Sketchingcurvatureforefficientout-of-distribution
| detectionfordeepneuralnetworks. |     | arXivpreprintarXiv:2102.12567,2021. |     |     |
| ------------------------------- | --- | ----------------------------------- | --- | --- |
[48] AndrewYKFoong,YingzhenLi,JoséMiguelHernández-Lobato,andRichardETurner. ’In-Between’
| UncertaintyinBayesianNeuralNetworks. |     | arXivpreprintarXiv:1906.11537,2019. |     |     |
| ------------------------------------ | --- | ----------------------------------- | --- | --- |
[49] AgustinusKristiadi,MatthiasHein,andPhilippHennig. LearnableUncertaintyunderLaplaceApproxima-
tions. InUAI,2021.
| [50] JoséMBernardoandAdrianFMSmith. |     | BayesianTheory. | JohnWiley&Sons,2009. |     |
| ----------------------------------- | --- | --------------- | -------------------- | --- |
[51] MatthiasHumt,JongseokLee,andRudolphTriebel.BayesianOptimizationMeetsLaplaceApproximation
forRoboticIntrospection. InIEEE/RSJInternationalConferenceonIntelligentRobotsandSystems(IROS)
Long-TermAutonomyWorkshop,2020.
[52] MarkNGibbs. BayesianGaussianProcessesforRegressionandClassification. Ph.D.Thesis,Department
ofPhysics,UniversityofCambridge,1997.
[53] DavidJCMacKay. ChoiceofBasisforLaplaceApproximation. Machinelearning,33(1),1998.
[54] PhilippHennig,DavidStern,RalfHerbrich,andThoreGraepel. KernelTopicModels. InAISTATS,2012.
[55] MariusHobbhahn,AgustinusKristiadi,andPhilippHennig. FastPredictiveUncertaintyforClassification
| withBayesianDeepNetworks. | arXivpreprintarXiv:2003.01227,2020. |     |     |     |
| ------------------------- | ----------------------------------- | --- | --- | --- |
[56] WesleyJMaddox,PavelIzmailov,TimurGaripov,DmitryPVetrov,andAndrewGordonWilson. Code
https://github.com/wjmaddox/swa_
| repo for"A Simple | Baseline forBayesian | Deep Learning". |     |     |
| ----------------- | -------------------- | --------------- | --- | --- |
gaussian,2019.
[57] AgustinusKristiadi.Last-layerLaplaceapproximationcodeexamples.https://github.com/wiseodd/
last_layer_laplace,2020.
[58] JongseokLeeandMatthiasHumt. OfficialCode:EstimatingModelUncertaintyofNeuralNetworksin
https://github.com/DLR-RM/curvature,2020.
SparseInformationForm,ICML2020.
[59] AdamPaszke,SamGross,FranciscoMassa,AdamLerer,JamesBradbury,GregoryChanan,TrevorKilleen,
ZemingLin,NataliaGimelshein,LucaAntiga,AlbanDesmaison,AndreasKopf,EdwardYang,Zachary
DeVito,MartinRaison,AlykhanTejani,SasankChilamkurthy,BenoitSteiner,LuFang,JunjieBai,and
SoumithChintala. PyTorch:AnImperativeStyle,High-PerformanceDeepLearningLibrary. InNeurIPS,
2019.
[60] KazukiOsawa. ASDL:Automaticsecond-orderdifferentiation(forfisher,gradientcovariance,hessian,
https://github.com/kazukiosawa/asdfghjkl,2021.
jacobian,andkernel)library.
[61] AshishVaswani,Noam Shazeer,Niki Parmar,JakobUszkoreit,Llion Jones,Aidan N Gomez,Lukasz
| Kaiser,andIlliaPolosukhin. | AttentionisAllYouNeed. |     | InNIPS,2017. |     |
| -------------------------- | ---------------------- | --- | ------------ | --- |
[62] Dan Hendrycks and Kevin Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution
| ExamplesinNeuralNetworks. | InICLR,2017. |     |     |     |
| ------------------------- | ------------ | --- | --- | --- |
[63] DanHendrycksandThomasDietterich. BenchmarkingNeuralNetworkRobustnesstoCommonCorrup-
| tionsandPerturbations. | InICLR,2019. |     |     |     |
| ---------------------- | ------------ | --- | --- | --- |
[64] YanivOvadia,EmilyFertig,JieRen,ZacharyNado,DavidSculley,SebastianNowozin,JoshuaDillon,
Balaji Lakshminarayanan,and Jasper Snoek. Can You Trust Your Model’s Uncertainty? Evaluating
| PredictiveUncertaintyunderDatasetShift. |     | InNeurIPS,2019. |     |     |
| --------------------------------------- | --- | --------------- | --- | --- |
13

[65] ZhiyunLu,EugeneIe,andFeiSha. UncertaintyEstimationwithInfinitesimalJackknife,ItsDistribution
andMean-FieldApproximation. arXivpreprintarXiv:2006.07584,2020.
[66] YemingWen,PaulVicol,JimmyBa,DustinTran,andRogerGrosse.Flipout:EfficientPseudo-Independent
WeightPerturbationsonMini-Batches. InICLR,2018.
[67] RuqiZhang,ChunyuanLi,JianyiZhang,ChangyouChen,andAndrewGordonWilson.CyclicalStochastic
GradientMCMCforBayesianDeepLearning. InICLR,2020.
[68] PangWeiKoh,ShioriSagawa,HenrikMarklund,SangMichaelXie,MarvinZhang,AkshayBalsubramani,
Weihua Hu,Michihiro Yasunaga,Richard Lanas Phillips,Irena Gao,et al. WILDS: A Benchmarkof
In-The-WildDistributionShifts. InarXivpreprintarXiv:2012.07421,2020.
[69] GaoHuang,ZhuangLiu,LaurensVanDerMaaten,andKilianQWeinberger. DenselyConnectedConvo-
lutionalNetworks. InCVPR,2017.
[70] VictorSanh,LysandreDebut,JulienChaumond,andThomasWolf. DistilBERT,aDistilledVersionof
Bert:Smaller,Faster,CheaperandLighter. In5thWorkshoponEnergyEfficientMachineLearningand
CognitiveComputing-NeurIPS,2019.
[71] VolodymyrKuleshov,NathanFenner,andStefanoErmon.AccurateUncertaintiesforDeepLearningUsing
CalibratedRegression. InICML,2018.
[72] OlivierChapelleandLihongLi. AnEmpiricalEvaluationofThompsonSampling. InNIPS,2011.
[73] MijungPark,GregHorwitz,andJonathanWPillow. ActiveLearningofNeuralResponseFunctionswith
GaussianProcesses. InNIPS,2011.
[74] IanJGoodfellow,MehdiMirza,DaXiao,AaronCourville,andYoshuaBengio.AnEmpiricalInvestigation
ofCatastrophicForgettinginGradient-BasedNeuralNetworks. arXivpreprintarXiv:1312.6211,2013.
[75] MichalisKTitsias,JonathanSchwarz,AlexanderGdeGMatthews,RazvanPascanu,andYeeWhyeTeh.
FunctionalRegularisationforContinualLearningwithGaussianProcesses. InICLR,2020.
[76] PingboPan,SiddharthSwaroop,AlexanderImmer,RunaEschenhagen,RichardETurner,andMoham-
madEmtiyazKhan.ContinualDeepLearningbyFunctionalRegularisationofMemorablePast.InNeurIPS,
2020.
[77] MaxWellingandYeeWTeh. BayesianLearningviaStochasticGradientLangevinDynamics. InICML,
2011.
[78] FlorianWenzel,KevinRoth,BastiaanSVeeling,JakubS´wia˛tkowski,LinhTran,StephanMandt,Jasper
Snoek,TimSalimans,RodolpheJenatton,andSebastianNowozin. HowGoodistheBayesPosteriorin
DeepNeuralNetworksReally? ICML,2020.
[79] PavelIzmailov,SharadVikram,MatthewDHoffman,andAndrewGordonWilson. WhatAreBayesian
NeuralNetworkPosteriorsReallyLike? InICML,2021.
[80] AdriàGarriga-AlonsoandVincentFortuin. Exactlangevindynamicswithstochasticgradients. arXiv
preprintarXiv:2102.01691,2021.
[81] AndrewGWilsonandPavelIzmailov. BayesianDeepLearningandaProbabilisticPerspectiveofGeneral-
ization. InNeurIPS,2020.
[82] AnoopKorattikara,VivekRathod,KevinMurphy,andMaxWelling. BayesianDarkKnowledge. InNIPS,
2015.
[83] Kuan-Chieh Wang,Paul Vicol,James Lucas,Li Gu,Roger Grosse,and Richard Zemel. Adversarial
DistillationofBayesianNeuralNetworkPosteriors. InICML,2018.
[84] RunaEschenhagen,ErikDaxberger,PhilippHennig,andAgustinusKristiadi. MixturesofLaplaceAp-
proximationsforImprovedPost-HocUncertaintyinDeepLearning. NeurIPSWorkshoponBayesianDeep
Learning,2021.
[85] JonathanFrankleandMichaelCarbin. TheLotteryTicketHypothesis:FindingSparse,TrainableNeural
Networks. InICLR,2019.
[86] DavidJCMacKay. APracticalBayesianFrameworkForBackpropagationNetworks. Neuralcomputation,
1992.
14

[87] SebastianFarquhar,LewisSmith,andYarinGal. LibertyorDepth:DeepBayesianNeuralNetsDoNot
| NeedComplexWeightPosteriorApproximations. |     | InNeurIPS,2020. |     |
| ----------------------------------------- | --- | --------------- | --- |
[88] ArjunKGuptaandDayaKNagar. MatrixVariateDistributions. ChapmanandHall,1999.
[89] CarlEckartandGaleYoung. Theapproximationofonematrixbyanotheroflowerrank. Psychometrika,1
(3):211–218,1936.
| [90] ChristopherM.Bishop. | PatternRecognitionandMachineLearning. |     | Springer,2006. |
| ------------------------- | ------------------------------------- | --- | -------------- |
[91] AnqiWu,SebastianNowozin,EdwardMeeds,RichardE.Turner,JoseMiguelHernandez-Lobato,and
AlexanderL.Gaunt. DeterministicVariationalInferenceforRobustBayesianNeuralNetworks. InICLR,
2019.
[92] AmrAhmedandEricPXing. SeekingTheTrulyCorrelatedTopicPosterior—OnTightApproximate
| InferenceofLogistic-NormalAdmixtureModel. |     | InAISTATS,2007. |     |
| ----------------------------------------- | --- | --------------- | --- |
[93] MichaelBraun andJon McAuliffe. VariationalInference forLarge-Scale Models ofDiscrete Choice.
JournaloftheAmericanStatisticalAssociation,105(489),2010.
[94] Yann LeCun,Léon Bottou,Yoshua Bengio,and Patrick Haffner. Gradient-based learning applied to
| documentrecognition.                   | ProceedingsoftheIEEE,86(11):2278–2324,1998. |                       |              |
| -------------------------------------- | ------------------------------------------- | --------------------- | ------------ |
| [95] SergeyZagoruykoandNikosKomodakis. |                                             | WideResidualNetworks. | InBMVC,2016. |
[96] IlyaLoshchilovandFrankHutter. SGDR:StochasticGradientDescentwithWarmRestarts. InICLR,
2017.
[97] RanganathKrishnanandPieroEsposito.Bayesian-Torch:BayesianNeuralNetworkLayersforUncertainty
Estimation. https://github.com/IntelLabs/bayesian-torch,2020.
[98] Florian Wenzel,Jasper Snoek,Dustin Tran,and Rodolphe Jenatton. Hyperparameter Ensembles for
| RobustnessandUncertaintyQuantification. |     | InNeurIPS,2020. |     |
| --------------------------------------- | --- | --------------- | --- |
[99] FerencHuszár.Noteonthequadraticpenaltiesinelasticweightconsolidation.ProceedingsoftheNational
AcademyofSciences,page201717042,2018.
15

| AppendixA |     | Derivation |     |     |     |     |     |     |     |     |     |
| --------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
A.1 TheDerivationoftheLaplaceApproximation
Letp(θ|D)beanintractableposterior,writtenas
|     |     |     |          |     | 1   |              |     | 1    |     |     |     |
| --- | --- | --- | -------- | --- | --- | ------------ | --- | ---- | --- | --- | --- |
|     |     |     | p(θ|D):= |     |     | p(D|θ)p(θ)=: |     | h(θ) |     |     | (1) |
(cid:82) p(D|θ)p(θ)dθ
Z
Our goal is to approximate this distribution with a Gaussian arising from the Laplace approxi-
mation. The key observation is that we can rewrite the normalizing constant Z as the integral
| (cid:82) exp(logh(θ))dθ. |     |     | θ := | argmax |     | logp(θ|D) | = argmax | logh(θ) |     |              |       |
| ------------------------ | --- | --- | ---- | ------ | --- | --------- | -------- | ------- | --- | ------------ | ----- |
|                          |     | Let | MAP  |        | θ   |           |          | θ       |     | be a (local) | maxi- |
mumoftheposterior—theso-calledmaximumaposteriori(MAP)estimate.Taylor-expandinglogh
| aroundθ | uptothesecondorderyields |     |     |     |     |     |     |     |     |     |     |
| ------- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MAP
1
|     |     |     | logh(θ)≈h(θ |     | )−  | (θ−θ | )(cid:62)Λ(θ−θ |     | ),  |     |     |
| --- | --- | --- | ----------- | --- | --- | ---- | -------------- | --- | --- | --- | --- |
|     |     |     |             |     | MAP |      | MAP            | MAP |     |     | (2) |
2
where Λ := −∇2logh(θ)| is the negative Hessian matrix ofthe log-jointin (1),evaluatedat
θ MAP
θ .Similartoitsoriginalformulation,hereweagainobtaina(multivariate)Gaussianintegral,the
MAP
analyticsolutionofwhichisreadilyavailable:
|     |     |     |     |     | (cid:90) | (cid:18) |     |     | (cid:19) |     |     |
| --- | --- | --- | --- | --- | -------- | -------- | --- | --- | -------- | --- | --- |
1
|     | Z   | ≈exp(logh(θ |     | ))  | exp | − (θ−θ | )(cid:62)Λ(θ−θ |     | )   | dθ  |     |
| --- | --- | ----------- | --- | --- | --- | ------ | -------------- | --- | --- | --- | --- |
|     |     |             |     | MAP |     |        | MAP            |     | MAP |     |     |
2
(3)
(2π)d
2
|     |     | =h(θ | )   |     | .   |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MAP (detΛ)1
2
Pluggingtheapproximations(2)and(3)backintotheexpressionofp(θ|D),weobtain
|     |         |     | 1     | (detΛ)1 |     | (cid:18) | 1     |                |     | (cid:19) |     |
| --- | ------- | --- | ----- | ------- | --- | -------- | ----- | -------------- | --- | -------- | --- |
|     |         |     |       |         | 2   |          |       | )(cid:62)Λ(θ−θ |     |          |     |
|     | p(θ|D)= |     | h(θ)≈ |         |     | exp −    | (θ−θ  |                |     | ) ,      | (4) |
|     |         |     | Z     | (2π)d   |     |          | 2 MAP |                | MAP |          |     |
2
which we can immediately identify as the Gaussian density N(θ|θ ,Σ) with mean θ and
|                   |     |                            |     |     |     |     |     | MAP |     | MAP |     |
| ----------------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| covariancematrixΣ |     | :=Λ−1.                     |     |     |     |     |     |     |     |     |     |
| AppendixB         |     | DetailsontheFourComponents |     |     |     |     |     |     |     |     |     |
1 InferenceoverSubsetsofWeights
B.1.1 Subnetwork
| StoringthefullD×D |     |     | covariancematrixΣ |     |     |     |     |     |     |     |     |
| ----------------- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
oftheweightposteriorinEq.(4)iscomputationally
intractableforamodernneuralnetworks.Oneapproachtoreducethiscomputationalburdenisto
performinferenceoveronlyasmallsubsetofthemodelparametersθ[27].Thisismotivatedbyrecent
findingsthatneuralnetscanbeheavilyprunedwithoutsacrificingtestaccuracy[85],andthatinthe
neighborhoodofalocaloptimum,therearemanydirectionsthatleavethepredictionsunchanged
[46].
ThissubnetworkinferenceapproachusesthefollowingapproximationtotheposteriorinEq.(4):
(cid:89)
|     |     |     | p(θ|D) | ≈ p(θ | S |D) | δ(θ | r −θ(cid:98)r ) = | q S (θ), |     |     | (5) |
| --- | --- | --- | ------ | ----- | ----- | --- | ----------------- | -------- | --- | --- | --- |
r
whereδ(x−a)denotestheDiracdeltafunctioncenteredata.Theapproximationq (θ)inEq.(5)
S
simplydecomposesthefullneuralnetworkposteriorp(θ|D)intoaLaplaceposteriorp(θ |D)over
S
thesubnetworkθ ∈RS,andfixed,deterministicvaluesθ(cid:98)r totheD−S remainingweightsθ .In
|     |     | S   |     |     |     |     |     |     |     |     | r   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
practice,theremainingweightsθ r aresimplysettotheirMAPestimates,i.e.θ(cid:98)r =θ MAP,requiringno
r
additionalcomputation.Importantly,notethatthesubnetworksizeS isinpracticeahyperparameter
thatcanbecontrolledbytheuser.Typically,S willbesetsuchthatthesubnetworkismuchsmaller
than the full network,i.e. S (cid:28) D. In particular,S can be set such that it is tractable to compute
×
and store the full S S covariance matrix over the subnetwork. This allows us to capture rich
16

dependenciesacrosstheweightswithinthesubnetwork.However,inprincipleonecouldalsoemploy
oneofthe(lessexpressive)factorizationsoftheHessian/FisherdescribedinSectionB.1.2.
Daxbergeret al. [27] propose to choose the subnetwork suchthat the subnetworkposteriorq (θ)
S
inEq.(5)isascloseaspossible(w.r.t.somediscrepancymeasure)tothefullposteriorp(θ|D)in
Eq.(4).AsthesubnetworkposteriorisdegenerateduetotheinvolvedDiracdeltafunctions,common
discrepancymeasuressuchastheKLdivergencearenotwelldefined. Therefore,Daxbergeretal.
[27]proposetousethesquared2-Wassersteindistance,whichinthiscasetakesthefollowingform:
(cid:18) (cid:16) (cid:17)1/2 (cid:19)
W (p(θ|D),q (θ))2 =Tr Σ+Σ −2 Σ 1/2 Σ Σ 1/2 , (6)
2 S S S S
wherethe(degenerate)subnetworkcovariancematrixΣ isequaltothefullcovariancematrixΣ but
S
withzerosatthepositionscorrespondingtotheweightsθ (i.e.thosenotpartofthesubnetwork).
r
Unfortunately,findingthesubsetofweightsθ ∈RS ofsizeS thatminimizesEq.(6)iscombinato-
S
riallyhard,asthecontributionofeachweightdependsoneveryotherweight.Daxbergeretal.[27]
thereforeassumethattheweightsareindependent,resultinginthefollowingsimplifiedobjective:
D
(cid:88)
W (p(θ|D),q (θ))2 ≈ σ2(1−m ), (7)
2 S d d
d=1
where σ2 = Σ is the marginalvariance ofthe dth weight,andm = 1 ifθ ∈ θ (withslight
d dd d d S
abuseofnotation)or0otherwiseisabinarymaskindicatingwhichweightsarepartofthesubnetwork
(seeDaxbergeretal.[27]fordetails).TheobjectiveinEq.(7)istriviallyminimizedbychoosinga
subnetworkcontainingtheS weightswiththehighestσ2values(i.e.withlargestmarginalvariances).
d
Inpractice,evencomputingthemarginalvariances(i.e.thediagonalofΣ)isintractable,asitrequires
storingandinvertingtheHessian/FisherΛ.Toapproximatethemarginalvariances,onecouldusea
diagonalLaplaceapproximation[43,2]thatassumesdiag(Σ)≈diag(Λ)−1.Alternatively,onecould
usediagonalSWAG[15].Formoredetailsonsubnetworkinference,refertoDaxbergeretal.[27].
B.1.2 Last-Layer
The last-layer Laplace [37, 28] is a special variant of the subnetwork Laplace where θ in (5) is
S
assumedtoequalthelast-layerweightmatrixW(L)ofthenetwork.Thatis,weletf :RM →RC
θ
isanL-layerNN,andassumethatthefirstL−1layersoff isafeaturemap.GivenMAP-trained
θ
parametersθ ,wedefineaLaplace-approximatedposterioroverW(L)
MAP
p(W(L)|D)≈N(W(L)|W (L) ,Σ(L)), (8)
MAP
andweleavetherestoftheparameterswiththeirMAP-estimatedvalues.Sincethismatrixissmall
relativetotheentirenetwork,thelast-layerLaplacecanbeimplementedefficiently.
2 HessianFactorization
Forbrevity,givenadatum(x,y),wedenotes(x,y)tobethegradientofthelog-likelihoodatθ ,
MAP
i.e.
s(x,y):=∇ p(y|f (x))| .
θ θ θ
MAP
Usingthisnotation,wecanwritetheFishercompactlyby
F := (cid:80)N E (s(x ,y)s(x ,y) (cid:124) ), (9)
n=1 p(y|fθ(xn)) n n
WeshallrefertothismatrixasthefullFisher.RecallthatF isaslargeastheexactHessianofthe
network,soitscomputationisofteninfeasible.Thus,here,wereviewseveralfactorizationschemes
thatmakesthecomputation(andstorage)oftheFisherefficient,startingfromthesimplest.
Diagonal Although MacKay recommended to not use the diagonal factorization of the Hessian
[86],arecentworkhasindicatedthisfactorizationisusableforsufficientlydeepNNs[87].Inthis
factorization,we simply assume that the negative-log-posterior’s Hessian Λ is simply a diagonal
matrixwithdiagonalelementsequalthediagonaloftheFisher,i.e.Λ≈−diag(F)(cid:62)I −λI.Since
wecanwritediag(F)= (cid:80)N E (s(x ,y)(cid:12)s(x ,y)),6thisfactorizationisefficient:
n=1 p(y|fθMAP (xn)) n n
NotonlydoesitrequireonlyavectoroflengthDtorepresentF butalsoitincursonlyaO(D)cost
wheninvertingΛ—downfromO(D3).
6Theoperator(cid:12)denotestheHadamardproduct.
17

KFAC The KFAC factorization can be seen as a midpoint between the two extremes: diagonal
factorization,whichmightbetoorestrictive,andthefullFisher,whichiscomputationallyinfeasible.
The key idea is to model the correlation between weights in the same layer but assume that any
pairofweightsfromtwodifferentlayersareindependent—thisisamoresophisticatedassumption
comparedtothediagonalfactorizationsincethere,itisassumedthatallweightsareindependentof
eachother.Foranylayerl=1,...,L,denotingN asthenumberofhiddenunitsatthel-thlayer,let
l
W(l) ∈RNl×Nl−1 betheweightmatrixofthel-thlayerofthenetwork,a(l) thel-thhiddenvector,
andg(l) ∈ RNl thelog-likelihoodgradientw.r.t.a(l).Foreachl = 1,...,L,wecanthenwritethe
outerproductinsideexpectationin(8)ass(x ,y)s(x ,y)(cid:62) =a(l−1)a(l)(cid:62)⊗g(l)g(l)(cid:62).Furthermore,
i i
assumingthata(l−1)isindependentofg(l),weobtaintheapproximationofthel-thdiagonalblockof
F,whichwedenotebyF(l):
(cid:16) (cid:17) (cid:16) (cid:17)
F(l) ≈E a(l−1)a(l−1)(cid:62) ⊗E g(l)g(l)(cid:62) =:A(l−1)⊗G(l), (10)
wherewerepresentboththesumandtheexpectationin(9)asEforbrevity.
From the previous expression we can see thatthe space complexityforstoring F(l) is reducedto
O(N2+N2 ),downfromO(N2N2 ).ConsideringallLlayersofthenetwork,weobtainthelayer-
l l−1 l l−1
wiseKroneckerfactors{A(l)}L−1and{G(l)}L ofthelog-likelihood’sHessian.Thiscorresponds
l=0 l=1
totheblock-diagonalapproximationofthefullHessian.
OnecanthenreadilyusetheseKroneckerfactorsinaLaplaceapproximation.Foreachlayerl,we
obtainthel-thdiagonalblockofΛ—denotedΛ(l)—by
(cid:16) √ (cid:17) (cid:16) √ (cid:17)
Λ(l) ≈ A(l−1)+ λI ⊗ G(l)+ λI
=:V(l)⊗U(l).
Note that we take the square root of the prior precision to avoid “double-counting” the effect of
the prior. Nonetheless, this can still be a crude approximation [19, 26]. This particular Laplace
approximation has been studied by Ritter et al. [23, 24] and can be seen as approximating
the posterior of each W(l) with the matrix-variate Gaussian distribution [88]: p(W(l)|D) ≈
MN(W(l)|W (l) ,U(l)−1,V(l)−1).Hence,samplingcanbedoneeasilyinalayer-wisemanner:
MAP
(cid:16) (cid:17)
W(l) ∼p W(l)|D ⇐⇒ W(l) =W (l) +U(l)−1 2EV(l)− 2 1
MAP
where
E ∼MN(0,I ,I ),
Nl Nl−1
wherewehavedenotedbyI theidentityb×bmatrix,forb∈N.Notethattheabovematrixinversions
b
andsquare-rootareingeneralmuchcheaperthanthoseinvolvingtheentireΛ.SamplingE isnot
aproblemeithersinceMN(0,I ,I )isequivalenttothestandard(N N )-variateNormal
Nl Nl−1 l l−1
distribution. As an alternative,Immeret al. [26] suggestto incorporate the priorexactly using an
eigendecompositionoftheindividualKroneckerfactors,whichcanimproveperformance.
Low-rank block-diagonal We can improve KFAC’s efficiency by considering its low-rank fac-
torization [29]. The key idea is to eigendecompose the Kronecker factors in (10) and keep only
the eigenvectors corresponding to the firstk largesteigenvalues. This can be done employing the
eigenvalue-correctedKFAC[44].Thatis,foreachlayerl=1,...,L:
(cid:16) (cid:17) (cid:16) (cid:17)
F(l) ≈ U (l−1) S (l−1) U (l−1)(cid:62) ⊗ U (l) Sl U (l)(cid:62)
A A A G G G
(cid:16) (cid:17)(cid:16) (cid:17)(cid:16) (cid:17)(cid:62)
= U
(l−1)⊗U (l)
S
(l−1)⊗S (l)
U
(l−1)⊗U (l)
.
A G A G A G
Underthisdecomposition,onecantheeasilyobtaintheoptimalrank-kapproximationofF(l),denoted
(l)
byF ,byselectingthetop-keigenvalues.However,thediagonalofthisrank-kmatrixcandeviate
k
too far from the exact diagonal elements of F(l). Hence,one can make the diagonal of this low
rankmatrixexactreplacingdiag(Fl)withdiag(F(l)),andobtainthefollowingrank-k-plus-diagonal
k
approximationofF(l):
F(l) ≈F (l) +diag(F(l))−diag(F (l) ).
k k
18

Thisfactorizationcanbeseenasacombinationoftheprevioustwoapproximations:Foreachdiagonal
blockofF,weusetheexactdiagonalelementsofF andapproximatetheoff-diagonalelementswith
arank-kmatrixarisingfromKFAC.Boththespaceandcomputationalcomplexitiesarelowerthan
thoseofKFACsincehereweworkexclusivelywithtruncatedanddiagonalmatrices.
Low-rank Insteadofonlyapproximatingeachblockbyalow-rankstructure,theentireHessian
orGGN can also be approximated by a low-rank structure [47, 46]. Eigendecomposition of F is
a convenient way to obtain a low-rank approximation. The eigendecomposition of F is given by
QLQ(cid:62)wherethecolumnsofQ∈RD×DareeigenvectorsofF andL=diag(l)isaD-dimensional
diagonalmatrixofeigenvalues.Assumingtheeigenvaluesinl arearrangedinadescendingorder,
the optimal k-rank approximation in Frobenius or spectral norm is given by truncation [89]: let
Q(cid:98) ∈RD×kbethematrixofthefirstkeigenvectorscorrespondingtothelargestkeigenvalues(cid:98)l∈Rk.
Thatis,wetruncatealleigenvectorsandeigenvaluesafterthek largesteigenvalues.Thelow-rank
approximationisthengivenby
F ≈Q(cid:98)diag((cid:98)l)Q(cid:98) (cid:62).
TherankkcanbechosenbasedontheeigenvaluessoastoretainasmuchinformationoftheHessian
(approximation)aspossible.Further,samplingandcomputationofthelog-determinantcanbecarried
outefficiently.
Functional Whenconsideringnetworklinearizationforthepredictivedistribution,wecandirectly
infertheGaussiandistributionontheoutputs,ofwhichtherearetypicallyfew,insteadofinferringa
distributionontheparameters,ofwhichtherearemany[25,26].
3 HyperparameterTuning
Inthissectionwefocusontuningthepriorvariance/precisionhyperparameterforsimplicity.Thesame
principlecanbeusedforotherhyperparametersoftheLaplaceapproximationsuchthatobservation
noiseinthecaseofregression.
Post-Hoc Here,weassumethatthestepsoftheLaplaceapproximation—MAPtrainingandforming
theGaussianapproximation—astwoindependentsteps.Assuch,wearefreetochoosedifferentprior
varianceγ2inthelatterpart,irrespectivetotheweightdecayhyperparameterusedintheformer.Here,
wereviewseveralwaystooptimizeγ2post-hoc.Ritteretal.[23]proposestotuneγ2bymaximizing
the posterior-predictive overa validation setD val := (x n ,y n )N n= val 1 . Thatis we solve the following
one-parameteroptimizationproblem:
N
(cid:88)val
γ2 =argmax logp(y |x ,D). (11)
∗ n n
γ2
n=1
However,Kristiadietal.[28]foundthatthepreviousobjectivetendstomaketheLaplaceapproxi-
mationoverconfidenttooutliers.Hence,theyproposedtoaddanauxiliarytermthatdependsonan
OODdatasetD out :=(x ( n out) )N n= ou 1 t to(11),asfollows
N N
(cid:88)val (cid:88)out (cid:104) (cid:105)
γ2 =argmax logp(y |x ,D)+λ H p(y |x(out),D) , (12)
∗ n n n n
γ2
n=1 n=1
whereH istheentropyfunctionalandλ∈(0,1]isatrade-offhyperparameter.Intuitively,wechoose
γ2 thatbalances the calibration on the true datasetandthe low-confidence on outliers. Moreover,
other losses could be constructed to tune the prior precision for optimal performance w.r.t. some
desiredquantity.Finally,inspiredbyImmeretal.[22](furtherdetailsbelowinOnline)onecanalso
maximizetheLaplace-approximatedmarginallikelihood(3)toobtainγ2,whicheliminatestheneed
∗
forthevalidationdata.
Online Contrary to the post-hoc tuning above, here we perform a Laplace approximation and
tune the prior variance simultaneously as we perform a MAP training [22]. The key is to form
a Laplace-approximated posteriorevery B epochs of a gradient descent,and use this posteriorto
approximatethemarginallikelihood,cf.(3).Bymaximizingthismarginallikelihood,wecanfind
thebesthyperparameters.Thus,oncetheMAPtraininghasfinished,weautomaticallyobtainaprior
variancethatisalreadysuitablefortheLaplaceapproximation.Notethat,thisway,onlyasingleMAP
trainingneedstobedone.Thisisincontrasttotheclassic,offlineevidenceframework[34]where
19

Algorithm1OnlineLaplace(adaptedfromImmeretal.[22,Algorithm1])
Input:
NNf ;trainingsetD;learningrateα andnumberofepochsT forMAPestimation;learning
|     | θ   |     |     | 0   |     |     | 0   |
| --- | --- | --- | --- | --- | --- | --- | --- |
rateα andnumberofepochsT forhyperparametertuning;marginallikelihoodmaximization
|     | 1   |     |     | 1   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
frequencyF.
| 1: Initializeθ  |         | 0        |      |     |     |     |     |
| --------------- | ------- | -------- | ---- | --- | --- | --- | --- |
| 2: fort=1,...,T |         | 0 do     |      |     |     |     |     |
| 3:              | g ←∇    | L(D;θ)|  |      |     |     |     |     |
|                 | t       | θ        | θt−1 |     |     |     |     |
| 4:              | θ ←θ    | −α       | g    |     |     |     |     |
|                 | t       | t−1 0    | t    |     |     |     |     |
|                 | ift mod | F =0then |      |     |     |     |     |
5:
|     | p(θ|D)≈N(θ|θ |     | ,(∇2L(D;θ)| |     | )−1) |     |                                      |
| --- | ------------ | --- | ----------- | --- | ---- | --- | ------------------------------------ |
| 6:  |              |     | t           |     | θt   |     | (cid:46)PerformaLaplaceapproximation |
7: for(cid:101)t=1,...,T do (cid:46)Hyperparameteroptimization
1
8: h ←∇ logp(D|γ2)| (cid:46)Themarginallikelihoodfollowsfrom(3)
|     |     | (cid:101)t | γ2  | γ 2 |     |     |     |
| --- | --- | ---------- | --- | --- | --- | --- | --- |
t(cid:101) −1
|     |        | γ 2 ←γ 2              | +α h         |     |     |     |     |
| --- | ------ | --------------------- | ------------ | --- | --- | --- | --- |
| 9:  |        |                       | 1 (cid:101)t |     |     |     |     |
|     |        | (cid:101)t (cid:101)t | −1           |     |     |     |     |
| 10: | endfor |                       |              |     |     |     |     |
11: endif
endfor
12:
| returnθ |     | ;∇2L(D;θ)| |     |     |     |     |     |
| ------- | --- | ---------- | --- | --- | --- | --- | --- |
| 13:     | T0  |            | θT0 |     |     |     |     |
themarginallikelihoodmaximizationisperformedonlywhentheMAPestimationisdone,andthese
stepsneedtobeiterativelydoneuntilconvergence.Asafinalnote,similartothepost-hocmarginal
likelihoodabove,thisonlineLaplacedoesnotrequireavalidationsetandhasanadditionalbenefitof
improvingthenetwork’sgeneralizationperformance[22].WereferthereadertoAlgorithm1foran
overview.
4 ApproximatePredictiveDistribution
Here,wedenotex ∈RN tobeatestpoint,andf bethenetworkoutputatthispoint.Wewillreview
|     |     | ∗   |     |     | ∗   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
differentwaytoapproximatethepredictivedistributionp(y|x ,D)givenaGaussianapproximate
∗
posterior,startingfromthemostgeneral.
B.4.3 General
MonteCarloIntegration
ThesimplestbutgeneralandunbiasedapproximationistheMonteCarlo
(MC)integration,whichcanbeperformedbysamplinganapproximateposteriorq(θ|D)repeatedly:
S
1 (cid:88)
|     |     | p(y|x | ,D)≈ | p(y|f | (x )), | whereθ | ∼q(θ|D). |
| --- | --- | ----- | ---- | ----- | ------ | ------ | -------- |
|     |     |       | ∗ S  |       | θs ∗   |        | s        |
s=1
√
While the error of this approximation decays like 1/ S and thus requires many samples to be
accurate,forpracticalBNNs,itisstandardtouse10or20samplesofq(θ|D)[23,28,12,etc.].Note
thatthisapproximationcanbeusedregardlesstheformofthelikelihoodp(y|f θ (x)),inparticularit
canbeusedtodirectlyobtainthepredictivedistributioninboththeregressionandclassificationalike.
B.4.4 DistributionofNetworkOutputs
Here, we are concerned in approximating the marginal distribution of f(x ), where θ has been
∗
integratedout.
| Linearization |     | Inthisapproximation,welinearizethenetworktoobtain |      |     |                    |     |     |
| ------------- | --- | ------------------------------------------------- | ---- | --- | ------------------ | --- | --- |
|               |     |                                                   | f (x | )≈f | (x )+J(cid:62)(θ−θ |     | ),  |
|               |     |                                                   | θ    | ∗ θ | MAP ∗ ∗            |     | MAP |
whereJ :=∇ f (x )| ∈Rd×c istheJacobianmatrixofthenetworkoutput.Thisway,under
|                                                                                  | ∗   | θ θ ∗ | θ   |     |     |     |     |
| -------------------------------------------------------------------------------- | --- | ----- | --- | --- | --- | --- | --- |
| aGaussianapproximateposteriorq(θ|D),themarginaldistributionoverthenetworkoutputf |     |       | MAP |     |     |     |     |
∗ :=
20

f(x )isagainaGaussian,givenby7
∗
(cid:90)
|     | p(f | ∗ |f θ (x | ∗ ),x ∗ ,D)= |      | δ(f ∗ | −f θ (x ∗ ))q(θ|D)dθ |     |     |     |
| --- | --- | --------- | ------------ | ---- | ----- | -------------------- | --- | --- | --- |
|     |     |           |              | ≈N(f | |f    | (x ),J(cid:62)ΣJ     | )   |     |     |
|     |     |           |              |      | ∗     | θ MAP ∗              | ∗ ∗ |     |     |
This approximation has been extensively used for small networks [34],but it has since gone out
of favor in deep learning due to its cost—the Jacobian J ∗ needs to be computed per input point.
Nevertheless,thisapproximationisstillusefulintheoreticalworksduetoitsanalyticalnature[28,49,
84].Moreover,inproblemswhereitcanbeefficientlyuseinpractice,itoffersabetterapproximation
thanMC-integral[26,48].Duetothelinearizationinthenetworkparameters,itisfurtherpossible
to obtain a functional prior in the form of a Gaussian process [25, 26]. This allows to perform
function-spaceinferenceasopposedtoweight-spaceinferencewhichisamenabletodifferentHessian
approximationsthanthosepointedoutaboveinSectionB.1.2,andis,forexample,usefulforcontinual
learning[76].
B.4.5 Regression
AssumethatwealreadyhaveaGaussianapproximationtop(f |x ,D) ≈ N(f |µ ,Σ )viathe
|     |     |     |     |     |     |     | ∗ ∗ | ∗ ∗ | ∗   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
linearizationabove.Inregression,westillneedtoincorporatetheobservationnoiseβ
encodedinthe
,βI)8tomakeprediction.Thiscanbeeasilydoneinanexact
| (usually)GaussianlikelihoodN(y |     |     | ∗ |f ∗ |     |     |     |     |     |     |
| ------------------------------ | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
manner:
(cid:90)
|     | p(y | ∗ |x ∗ )= |     | N(y ∗ |f | ∗ ,βI)N(f | ∗ |µ | ∗ ,Σ ∗ )df ∗ |     |     |
| --- | --- | --------- | --- | -------- | --------- | ---- | ------------ | --- | --- |
RC
|     |     |     | =N(y | |µ ,Σ | +βI), |     |     |     |     |
| --- | --- | --- | ---- | ----- | ----- | --- | --- | --- | --- |
|     |     |     |      | ∗ ∗   | ∗     |     |     |     |     |
sincetheintegralaboveisjustaconvolutionoftwoGaussianr.v.s.
B.4.6 ClassificationandGeneralizedRegression
Sinceunliketheregressioncase,theclassificationlikelihoodp(y |f )isnon-Gaussian,wecannot
∗ ∗
analyticallyobtainp(y |x )givenaGaussianapproximationp(f |x ,D)≈N(f |µ ,Σ ).So,in
|     | ∗   | ∗   |     |     |     |     | ∗ ∗ | ∗   | ∗ ∗ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thiscaseweareinterestedinapproximatingtheintractableintegral
(cid:90)
|     |     | p(y |x | )=  | p(y | |f )N(f | |µ ,Σ | )df , |     |     |
| --- | --- | ------ | --- | --- | ------- | ----- | ----- | --- | --- |
|     |     | ∗      | ∗   | ∗   | ∗       | ∗ ∗   | ∗ ∗   |     |     |
wherep(y |f )isconstructedviaaninverse-linkfunction.Herewewillreviewtheusualcaseof
| ∗ ∗ |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
classification,i.e.whenp(y |f )=σ(f )whereσisthelogistic-sigmoidfunction,orp(y |f )=
|     |     | ∗ ∗ |     | ∗   |     |     |     |     | ∗ ∗ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
softmax(f ).
∗
DeltaMethod Thecruxofthedeltamethod[91–93]isaTaylor-expansionofthesoftmaxfunction
aroundµ uptothesecondorder.Then,sincep(f |x ,D)isassumedtobeGaussian,theintegral
| ∗   |     |     |     |     | ∗   | ∗   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E (softmax(f ))canbecomputedeasily,resultinginananalyticexpressionsoftmax(µ )+
| p(f∗|x∗,D)                                           | ∗   |     |     |     |     |     |     |     | ∗   |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1/2tr(BΣ ∗ ),whereBistheHessianmatrixofthesoftmaxatµ |     |     |     |     |     |     | ∗ . |     |     |
ProbitApproximations Theessenceofthe(binary)probitapproximation[31,34]istoapproxi-
mateσwiththeprobitfunctionΦ—thestandardNormalc.d.f.—whichmakestheintegralsolvable
analytically.Usingthisapproximation,onecanthenobtaintheclosed-formapproximation
(cid:90)
|     |     | p(y | |x )≈ | Φ(f | )N(f | |µ ,σ2)df |     |     |     |
| --- | --- | --- | ----- | --- | ---- | --------- | --- | --- | --- |
|     |     | ∗   | ∗     |     | ∗    | ∗ ∗       | ∗ ∗ |     |     |
R
|     |     |     |     | (cid:32) |     | (cid:33) |     |     |     |
| --- | --- | --- | --- | -------- | --- | -------- | --- | --- | --- |
µ ∗
|     |     |     | =σ  |           |     | .   |     |     |     |
| --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- |
|     |     |     |     | (cid:112) | π   | σ2  |     |     |     |
1+
|     |     |     |     |     | 8   | ∗   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Ithasageneralizationtomulti-classclassification,duetoGibbs[52],i.e.forapproximating
(cid:90)
|     |     | p(y |x )= |     | softmax(f | )N(f | |µ  | ,Σ )df . |     | (13) |
| --- | --- | --------- | --- | --------- | ---- | --- | -------- | --- | ---- |
|     |     | ∗ ∗       |     |           | ∗    | ∗   | ∗ ∗ ∗    |     |      |
RC
7SeeBishop[90,Sec.4.5.2].
| 8Weassumeamultivariateoutputy |     |     | ∈RC |                    |     |     |     |     |     |
| ----------------------------- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- |
|                               |     |     | ∗   | forfullgenerality. |     |     |     |     |     |
21

|     | testloglikelihood |     | testaccuracy |     | OOD-AUROC   | predictiontime(s) |     |
| --- | ----------------- | --- | ------------ | --- | ----------- | ----------------- | --- |
|     | -0.302±0.005      |     | 0.894±0.002  |     | 0.832±0.011 | 29.5±0.2          |     |
DIAG
|     | -0.282±0.004 |     | 0.899±0.002 |     | 0.836±0.004 | 30.6±0.1 |     |
| --- | ------------ | --- | ----------- | --- | ----------- | -------- | --- |
KFAC
| FULL | -0.285±0.004 |     | 0.898±0.002 |     | 0.876±0.003 | 62.8±1.1 |     |
| ---- | ------------ | --- | ----------- | --- | ----------- | -------- | --- |
Table2:QualitativecomparisonofdifferentHessianapproximations.TheKFACHessianapproxima-
tionperformssimilartoFULLGauss-NewtonbutisalmostasfastasDIAG.Weuseonlinemarginal
likelihoodmethod[22]totrainasmallconvolutionalnetworkonFMNISTandmeasureperformance
attesttime.Werepeatforthreeseedstoestimatethestandarderror.TheOOD-AUROCisaveraged
overEMNIST,MNIST,andKMNIST. Thepredictiontimeistakenastheaverageoverallinand
out-of-distributiondatasets.WeusetheMCpredictivewith100samples.
Inthiscase,weapproximatetheresultingprobabilityvectoroflengthC withavectorwhichi-th
(cid:112)
component is given by exp(τ )/ (cid:80)C exp(τ ), where τ = µ / 1+π/8Σ for each j =
|     |     | i   | j=1 | j   | j   | ∗j  | ∗jj |
| --- | --- | --- | --- | --- | --- | --- | --- |
1,...,C. This approximation ignores the correlation between logits since it only depends on the
diagonalofΣ .Nevertheless,ityieldsgoodresultsevenindeeplearning[65],andareinvaluable
∗
toolsfortheoreticalwork[84].
Laplace Bridge The main idea ofthe Laplace bridge is to perform a Laplace approximation to
overRC
the Dirichletdistribution byfirstwriting itas a distribution withthe helpofthe softmax
function[53,54].Thisway,Laplaceapproximationcanbereasonablyappliedtoapproximatethe
Dirichlet,whichcan be thoughtas mapping the DirichletDir(α ) to a Gaussian N(µ ,Σ ). The
|     |     |     |     |     |     | ∗   | ∗ ∗ |
| --- | --- | --- | --- | --- | --- | --- | --- |
pseudo-inverseofthismap,mapping(µ ,Σ )toα whereforeachi=1,...,C,thei-thcomponent
|     |     |     | ∗   | ∗   | ∗   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
αisgivenbythesimpleclosed-formexpression
|     |     |     |    |     |     |    |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
C
|     |     | 1   |     | 2 exp | ( µ i )(cid:88) |            |     |
| --- | --- | --- | --- | ----- | --------------- | ---------- | --- |
|     |     | α = | 1− | +     |                 | exp(−µ ), |     |
|     |     | i Σ |     | C     | C 2             | j          |     |
ii
j=1
istheLaplacebridge.Justliketheprobitapproximation,theLaplacebridgeignoresthecorrelation
|     |     |     |     |     |     | full | distribution |
| --- | --- | --- | --- | --- | --- | ---- | ------------ |
between logits. But, unlike all the previous approximations, it yields a over the
solutionsofthesoftmax-Gaussianintegral(13).So,theLaplacebridgeisaricheryetcomparably
simpleapproximationtotheintegralandisusefulformanyapplicationsindeepBNNs[55].
AppendixC FurtherExperimentsDetailsandResults
C.1 LaplaceComparison
Here,wepresentmoredetailedresultsofourcomparisonofthedifferentvariationsoftheLaplace
approximation. We show in-distribution accuracy for CIFAR-10 using a model trained with and
withoutdataaugmentation,andAUROCvaluesaveragedovertheout-of-distributiondatasetsSVHN,
LSUN,andCIFAR-100.InthefirstrowofFigure8,wehighlightthedifferentHessianstructureswith
differentcolors;inthesecondrow,weusecolortohighlightthedifferentlinkapproximationsinthe
predictivedistribution.Weconsideredmostcombinationsofthedifferentchoicesforthecomponents
discussedinSection2,butexcludesomecombinationswhichwehavefoundtonotworkwellatall,
e.g.onlineLaplacewhenperformingaLaplaceapproximationovertheweightsofonlythelastlayer.
InTable2,wecomparethepredictiveperformanceandruntimewhenusingdifferentlystructured
Hessian approximations. We findthatthe Kronecker-factoredHessian approximations provides a
goodtrade-offbetweenruntimeandperformance.
C.2 PredictiveUncertaintyQuantification
C.2.1 TrainingDetails
WeuseLeNet[94]andWideResNet-16-4[WRN,95]architecturesfortheMNISTandCIFAR-10
experiments, respectively. We adopt the commonly-used training procedure and hyperparameter
values.
22

0.92
0.90
0.88
0.88 0.89 0.90 0.91 0.92 0.93
Acc. (ID)
CORUA
0.9
MAP
hessian_structure
diag 0.8
kron
full
subset_of_weights 0.7
all
last_layer
inference_method
online 0.6
post-hoc
0.60 0.65 0.70 0.75 0.80 0.85 0.90
Acc. (ID)
(a)Hessianstructure(CIFAR-10+DA)
CORUA
(b)Hessianstructure(CIFAR-10)
0.92
0.90
0.88
0.88 0.89 0.90 0.91 0.92 0.93
Acc. (ID)
CORUA
0.9
MAP
link_approx
mc probit 0.8
bridge
map
subset_of_weights 0.7
all
last_layer
inference_method
0.6
online
post-hoc
0.60 0.65 0.70 0.75 0.80 0.85 0.90
Acc. (ID)
(c)Predictiveapproximation(CIFAR-10+DA)
CORUA
(d)Predictiveapproximation(CIFAR-10)
Figure8:ComparisonofvariationsoftheLAontheCIFAR-10OODexperimentwith((a)and(c))
andwithout((b)and(d))dataaugmentation(DA).
MAP WeuseAdamandNesterov-SGDtotrainLeNetandWRN,respectively.Theinitiallearning
rateis0.1andannealedviathecosinedecaymethod[96]over100epochs.Theweightdecayisset
to5×10−4.Unlessstatedotherwise,allmethodsbelowusethesetrainingparameters.
DE WetrainfiveMAPnetwork(seeabove)independentlytoformtheensemble.
VB WeusetheBayesian-Torchlibrary[97]totrainthenetwork.Thavariationalposteriorischosen
tobethediagonalGaussian[11,12]andtheflipoutestimator[66]isemployed.Thepriorprecisionis
setto5×10−4tomatchtheMAP-trainednetwork,whiletheKL-termdownscalingfactorissetto
0.1,following[13].
CSGHMC Weusethepubliclyavailablecodeprovidedbytheoriginalauthors[67].9 Weusetheir
default(i.e.recommended)hyperparameters.
SWAG FortheSWAGbaseline,wefollowMaddoxetal.[15]andrunstochasticgradientdescent
withaconstantlearningrateonthepre-trainedmodelstocollectonemodelsnapshotperepoch,fora
totalof40snapshots.Attesttime,wethenmakepredictionsbyusing30MonteCarlosamplesfrom
theposteriordistribution;wecorrectthebatchnormalizationstatisticsofeachsampleasdescribedin
Maddoxetal.[15].Totunetheconstantlearningrate,weusedthesameapproachasinEschenhagen
etal.[84],combiningagridsearchwithathresholdonthemeanconfidence.ForMNIST,wedefined
thegridtobetheset{1e-1,5e-2,1e-2,5e-3,1e-3},yieldinganoptimalvalueof1e-2.ForCIFAR-10,
searchingoverthesamegridsuggestedthattheoptimalvalueliesbetween5e-3and1e-3;another,
finer-grainedgridsearchovertheset{5e-3,4e-3,3e-3,2e-3,1e-3}thenrevealedthebestvalueto
be2e-3.
Otherbaselines Ourchoiceofbaselinesisbasedonthemostcommonandbestperformingmethods
ofrecentBayesianDLpapers.Despiteitspopularity,MonteCarlo(MC)dropout[6]hasbeenshown
tounderperformcomparedtomorerecentmethods(seee.g.Ovadiaetal.[64]).ArecentVImethod
calledVariationalOnlineGauss-Newton(VOGN)[13]alsoseemstounderperform.Forexample,
Fig.5ofOsawaetal.[13]showsthatonOODdetectionwithCIFAR-10vs.SVHN,MC-dropoutand
9https://github.com/ruqizhang/csgmcmc
23

1.0
0.6
0.75
1.0
|     |     |     | 0.4 |     |      | MAP | SWAG |
| --- | --- | --- | --- | --- | ---- | --- | ---- |
|     | 0.5 |     |     |     | 0.50 | DE  | LA   |
0.5
|     |     |     | 0.2 |     | 0.25 | BBB | LA* |
| --- | --- | --- | --- | --- | ---- | --- | --- |
CSGHMC
| 0.0      | 0.0   |        | 0.0   |         | 0.00 |     |       |
| -------- | ----- | ------ | ----- | ------- | ---- | --- | ----- |
| 0 50 100 | 150 0 | 50 100 | 150 0 | 1 2 3 4 | 5 0  | 1 2 | 3 4 5 |
(a)MNIST-RBrier↓ (b)MNIST-RAcc.↑ (c)CIFAR10-CBrier↓ (d)CIFAR10-CAcc.↑
Figure9:DatasetshiftontheRotated-MNIST(top)andCorrupted-CIFAR-10datasets(bottom).
Table3:MNISTOODdetectionresults.
|         |          | Confidence↓ |          |          | AUROC↑   |          |     |
| ------- | -------- | ----------- | -------- | -------- | -------- | -------- | --- |
| Methods | EMNIST   | FMNIST      | KMNIST   | EMNIST   | FMNIST   | KMNIST   |     |
| MAP     | 83.6±0.3 | 64.2±0.5    | 77.3±0.3 | 93.5±0.3 | 98.9±0.0 | 97.0±0.1 |     |
| DE      | 75.8±0.2 | 55.4±0.4    | 65.9±0.3 | 95.1±0.0 | 99.2±0.0 | 98.3±0.0 |     |
| BBB     | 79.1±0.4 | 67.5±1.6    | 73.1±0.4 | 92.3±0.2 | 98.2±0.2 | 97.0±0.2 |     |
| CSGHMC  | 76.2±1.6 | 63.6±1.9    | 67.9±1.5 | 93.4±0.2 | 97.7±0.2 | 97.1±0.1 |     |
| SWAG    | 64.9±0.3 | 84.0±0.2    | 78.5±0.3 | 98.9±0.0 | 93.6±0.3 | 97.1±0.1 |     |
| LA      | 74.8±0.4 | 58.8±0.5    | 69.0±0.4 | 93.4±0.3 | 98.5±0.1 | 96.6±0.1 |     |
| LA*     | 62.0±0.5 | 49.6±0.6    | 56.7±0.5 | 94.3±0.2 | 98.3±0.1 | 96.6±0.2 |     |
VOGNonlyachieveAUROC↑valuesof81.9and80.0,respectively,whilelast-layer-LAobtainsa
substantiallybettervalueof91.9(theyuseResNet-18,whichiscomparabletoourmodel).
C.2.2 DetailedResults
WeshowtheBrierscoreandaccuracyasafunctionofshiftintensityinFig.9.Moreover,weprovide
thedetailed(i.e.non-averaged)OODdetectionresultsinTables3and4.
C.2.3 AdditionalDetailsonWall-clockTimeComparison
Concerning the wall-clock time comparison in Fig. 5, we would like to clarify that for LA, we
considerthedefaultconfigurationoflaplace.AsthedefaultLAvariantusestheclosed-formprobit
approximationtothepredictivedistributionandthereforeneitherrequiresMonteCarlo(MC)sampling
normultipleforwardpasses,thewall-clocktimeformakingpredictionsisessentiallythesameasfor
MAP.Thisiscontrasttothebaselinemethods,whicharesignificantlymoreexpensiveatprediction
time due to the need for MC sampling (VB, SWAG) or forward passes through multiple model
snapshots(DE,CSGHMC).
Importantly,notethatisanadvantageexclusivetoourimplementationofLA(i.e.withaGGN/Fisher
Hessianapproximationorwiththelast-layerLA)thatitcanbeusedwithoutsampling(i.e.usingthe
probitorLaplacebridgepredictiveapproximations).Thiskindofapproximationisincompatiblewith
theotherbaselines(i.e.DE,CSGHMC,SWAG,andVB)sincethesemethodsjustyieldsamples/dis-
tributionsoverweightswhileourLAvariantsimplicitlyyieldaGaussiandistributionoverlogitsdue
tothelinearizationoftheNNinducedbytheuseoftheGGN/Fisher(seeImmeretal.[26]fordetails)
ortheuseofonlythelastlayer.Whileonecouldstillapplylinearizationtoothermethods,thiswould
notbetheoreticallyjustified,incontrasttoGGN-/last-layer-LA.
Finally, the reason we benchmark our deterministic, probit-based version is that we found it to
consistentlyperformonparorbetterthanMCsampling.IfwepredictwiththeLAusingMCsamples
on the logits,the runtime is onlyaround20% slowerthan the deterministicprobitapproximation,
whichisstillsignificantlyfasterthanallothermethods.
Insummary,webelievethattheabilitytoobtaincalibratedpredictionswithasingleforward-pass
is a criticalanddistinctive advantage ofthe LA overalmostallotherBayesian deeplearning and
ensemblemethods.
24

Table4:CIFAR-10OODdetectionresults.
Confidence↓ AUROC↑
Methods SVHN LSUN CIFAR-100 SVHN LSUN CIFAR-100
MAP 77.5±2.9 71.3±0.6 79.3±0.1 91.8±1.2 94.5±0.2 90.1±0.1
DE 62.8±0.7 62.6±0.4 70.8±0.0 95.4±0.2 95.3±0.1 91.4±0.1
BBB 60.2±0.7 53.8±1.1 63.8±0.2 88.5±0.4 91.9±0.4 84.9±0.1
CSGHMC 69.8±0.8 65.2±0.8 73.1±0.1 91.2±0.3 92.6±0.3 87.9±0.1
SWAG 69.3±4.0 62.2±2.3 73.0±0.4 91.6±1.3 94.0±0.7 88.2±0.5
LA 70.6±3.2 63.8±0.5 72.6±0.1 92.0±1.2 94.6±0.2 90.1±0.1
LA* 58.0±3.1 50.0±0.5 59.0±0.1 91.9±1.3 95.0±0.2 90.2±0.1
C.3 WILDSExperiments
Forthissetofexperiments,weuseWILDS[68],arecentlyproposedbenchmarkofrealisticdistribution
shiftsencompassingavarietyofreal-worlddatasetsacrossdifferentdatamodalitiesandapplication
domains.Inparticular,weconsiderthefollowingWILDSdatasets:
• Camelyon17:Tumorclassification(binary)ofhistopathologicaltissueimagesacrossdiffer-
enthospitals(IDvs.OOD)usingaDenseNet-121model(10seeds).
• FMoW:Building/landuseclassification(62classes)ofsatelliteimagesacrossdifferenttimes
andregions(IDvs.OOD)usingaDenseNet-121model(3seeds).
• CivilCommments:Toxicityclassification(binary)ofonlinetextcommentsacrossdifferent
demographicidentities(IDvs.OOD)usingaDistilBERT-base-uncasedmodel(5seeds).
• Amazon:Sentimentclassification(5classes)ofproductreviewsacrossdifferentreviewers
(IDvs.OOD)usingaDistilBERT-base-uncasedmodel(3seeds).
• PovertyMap: Asset wealth index regression (real-valued) across different countries and
rural/urbanareas(IDvs.OOD)usingaResNet-18model(5seeds).
Pleaserefertotheoriginalpaperformoredetailsonthisbenchmarkandtheabove-mentioneddatasets.
AllreportedresultsinFig.6andFig.10showthemeanandstandarderroracrossasmanyseedsas
thereareprovidedwiththeoriginalpaper(seethelistofdatasetsabovefortheexactnumbers).
Forthelast-layerLaplacemethod,weuseeitheraKFACorfullcovariancematrix(dependingonthe
sizeofthelastlayer;inparticular,weuseaKFACcovarianceforFMoWandfullcovariancesforall
otherdatasets)andthelinearizedMonteCarlopredictivedistributionwith10,000samples.
For the deep ensemble,we simply the aggregate the pre-trained models provided by the original
paper10 Thisyieldsensemblesof5neuralnetworkmodels,whichisacommly-usedensemblesize
[64].Sincethesemodelsweretrainedindifferentways(e.g.usingdifferentdomaingeneralization
methods,see[68]fordetails),theircombinationscanbeviewedashyperparameterensembles[98].
Notethatthetemperaturescalingbaselineisonlyapplicableforclassificationtasks,andthereforewe
donotreportitforthePovertyMapregressiondataset.
Wetunethetemperatureparameterfortemperaturescaling,thepriorprecisionparameterforLaplace,
andthenoisestandarddeviationparameterforregression(i.e.forthePovertyMapdataset)bymini-
mizingthenegativelog-likelihoodonthein-distributionvalidationsetsprovidedwithWILDS.
Finally, Fig. 10 shows an extended version of the results reported in Fig. 6, which additionally
reports the following metrics: accuracy(forclassification) ormean squarederror(forregression),
confidence(onlyforclassification),meancalibrationerror(onlyforclassification),andBrierscore
(onlyforclassification).TheoverallconclusionhereisthesameasforFig.6,namelythatLaplaceis
significantlybettercalibratedthanMAP,andcompetitivewithtemperaturescalingandensembles,
especially on the OOD splits. Note that the differences in accuracies of the ensemble stem from
the differenttraining procedures ofthe ensemble members (whichsometimes achieve higherand
sometimesloweraccuracy),asmentionedabove.
10See https://worksheets.codalab.org/worksheets/0x52cea64d1d3f4fa89de326b4e31aa50a
forthecompletelistofmodels.
25

|     |        | MAP    | DeepEnsemble | Temp. Scaling |     | Laplace |        |
| --- | ------ | ------ | ------------ | ------------- | --- | ------- | ------ |
|     | ID OOD | ID OOD | ID           | OOD           | ID  | OOD     | ID OOD |
0.65
| ESM/ycaruccA |     |     | 0.96 |      |     |     |      |
| ------------ | --- | --- | ---- | ---- | --- | --- | ---- |
|              |     |     |      | 0.74 |     |     | 0.22 |
0.9
0.60
0.94
| 0.8 |     |      |      | 0.73 |     |     | 0.20 |
| --- | --- | ---- | ---- | ---- | --- | --- | ---- |
|     |     | 0.55 | 0.92 |      |     |     |      |
| 0.7 |     |      |      | 0.72 |     |     | 0.18 |
0.90
1.00
0.95
|     |     | 0.8 | 0.95 | 0.825 |     |     |     |
| --- | --- | --- | ---- | ----- | --- | --- | --- |
ecnedfinoC
0.90
|     |     |     | 0.90 | 0.800 |     |     |     |
| --- | --- | --- | ---- | ----- | --- | --- | --- |
0.7
| 0.85 |     |     |     | 0.775 |     |     |     |
| ---- | --- | --- | --- | ----- | --- | --- | --- |
0.85
0.6
0.750
0.80
0.80
|     |     |     | 0.30 |     |     |     | 0.700 |
| --- | --- | --- | ---- | --- | --- | --- | ----- |
0.8
|     |     | 3.0 |      | 0.70 |     |     |       |
| --- | --- | --- | ---- | ---- | --- | --- | ----- |
|     |     |     | 0.25 |      |     |     | 0.675 |
0.6
| LLN |     | 2.5 | 0.20 |     |     |     | 0.650 |
| --- | --- | --- | ---- | --- | --- | --- | ----- |
0.65
| 0.4 |     |     | 0.15 |      |     |     |       |
| --- | --- | --- | ---- | ---- | --- | --- | ----- |
|     |     | 2.0 |      |      |     |     | 0.625 |
| 0.2 |     |     | 0.10 |      |     |     |       |
|     |     | 1.5 |      | 0.60 |     |     | 0.600 |
0.125
| 0.20 |     | 0.4 |     |     |     |     | 40  |
| ---- | --- | --- | --- | --- | --- | --- | --- |
0.3
| .bilaC/ECE |     |     |     | 0.100 |     |     |     |
| ---------- | --- | --- | --- | ----- | --- | --- | --- |
| 0.15       |     | 0.3 |     |       |     |     | 30  |
|            |     |     | 0.2 | 0.075 |     |     |     |
| 0.10       |     | 0.2 |     |       |     |     | 20  |
0.050
|     |     | 0.1 | 0.1 |     |     |     | 10  |
| --- | --- | --- | --- | --- | --- | --- | --- |
0.05
0.025
0.0
0.6
| 0.35 |     |     | 0.6 |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- |
0.5
| 0.30 |     | 0.4 |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- |
| ECM  |     |     | 0.4 |     |     |     |     |
0.25
0.4
0.2
| 0.20 |     |     | 0.2 |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- |
0.3
0.5
0.4
0.40
| 0.4 |     | 0.7 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
erocSreirB
0.3
| 0.3 |     |     |     | 0.38 |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- |
0.6
0.2
|     |     |     | 0.2 | 0.36 |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- |
0.1
0.5
(a) Camelyon17 (b) FMoW (c) CivilComments (d) Amazon (e) PovertyMap
Figure10:Assessingreal-worlddistributionshiftrobustnessonfivedatasetsfromtheWILDSbench-
mark [68],covering different data modalities,model architectures,and output types; see text for
details. Wereportmeans±standarderrorsofseveralmetrics(fromtoptobottom): accuracy(for
classification)ormeansquarederror(forregression),confidence(onlyforclassification),negative
log-likelihood,ECE(forclassification)orregressioncalibrationerror[71],meancalibrationerror
(onlyforclassification),andBrierscore(onlyforclassification).Thein-distribution(leftpanels)and
OOD(rightpanels)datasetsplitscorrespondtodifferentdomains(e.g.hospitalsforCamelyon17).
26

C.4 FurtherDetailsontheContinualLearningExperiment
WebenchmarkLaplaceapproximationsintheBayesiancontinuallearningsettingonthepermuted
MNIST benchmarkwhichconsistsof10consecutivetaskswhereeachtaskisapermutationofthe
pixelsoftheMNISTimages.Followingcommonpractice[24,7,13],weusea2-hiddenlayerMLP
with100hiddenunitseachand28×28=784inputdimensionsand10outputdimensionsforthe
MNISTclasses.WeadopttheimplementationofthecontinuallearningtaskandthemodelbyPanetal.
[76].11Inthefollowing,wewillbrieflyoutlinetheBayesianapproachtocontinuallearning[7]and
explainhowadiagonalandKFACLaplaceapproximationcanbeemployedinthissetting.Further,we
describehowthiscanbecombinedwiththeevidenceframeworktoupdatetheprioronlinealleviating
theneedforavalidationset,whichisunlikelytobeavailableinrealcontinuallearningscenarios.
C.4.1 BayesianApproachtoContinualLearning
The Bayesian approach to continual learning can be simply described as iteratively updating the
}T
posterioraftereachtask.WearegivenT datasetsD := {D andhaveaneuralnetworkwith
t t=1
parametersθ.InlinewiththestandardsupervisedlearningsettingoutlinedinSection2,wehavea
prioronparametersp(θ)=N(θ;0,γ2I)andalikelihoodp(D|θ)realizedbyaneuralnetwork.The
posteriorontheparametersafteralltasksisthen
|     |     | p(θ|D)∝p(D |     | |θ)×...×p(D | |θ)×p(D |           | |θ)×p(θ).          |           | (14) |
| --- | --- | ---------- | --- | ----------- | ------- | --------- | ------------------ | --------- | ---- |
|     |     |            |     | T           | 2       |           | 1                  |           |      |
|     |     |            |     |             |         | (cid:124) | (cid:123)(cid:122) | (cid:125) |      |
∝p(θ|D1)
|     |     |     |     |     | (cid:124) | (cid:123)(cid:122) |     | (cid:125) |     |
| --- | --- | --- | --- | --- | --------- | ------------------ | --- | --------- | --- |
∝p(θ|D1,D2)
Thisfactorizationgivesrisetoarecursiontoupdatetheposterioraftert−1datasetstotheposterior
aftertdatasets:
|     |     | p(θ|D | ,...,D | )∝p(D | |θ)p(θ|D | ,...,D |     | ).  | (15) |
| --- | --- | ----- | ------ | ----- | -------- | ------ | --- | --- | ---- |
|     |     |       | 1      | t     | t        | 1      | t−1 |     |      |
ThenormalizerforeachupdateinEq.(15)isgivenbythemarginallikelihoodp(D |D ,...,D )
|     |     |     |     |     |     |     |     | t 1 | t−1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andwewilluseitforoptimizingthevarianceγ2 ofp(θ). Incorporatinganewtaskisthesameas
Bayesian inference in the supervisedcase butwithan updatedprior,i.e.,the prioris the previous
posterior distribution on θ. The Laplace approximation provides one way to approximately infer
the posterior distributions after each task [99, 24, 76]. Alternatively,variational inference can be
used[7,13].
C.4.2 TheLaplaceApproximationforContinualLearning
TheLaplaceapproximationfacilitatestherecursiveupdates(Eq.(15))thatariseincontinuallearning.
Inthiscontext,itwasfirstsuggestedwithadiagonalHessianapproximationbyKirkpatricketal.[2,
EWC]andHuszár[99]correctedtheirupdates.Ritteretal.[24]greatlyimprovedtheperformance
byusingaKFACHessianapproximationinsteadofadiagonal.TheLaplaceapproximationtothe
|     |     |     |     |     | N(θ (t) ,Σ(t)) |     |     |     |     |
| --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- |
posterior after observing task t is a Gaussian We obtain θ MAP by optimizing the
MAP
unnormalizedlogposteriordistributiononθasannotatedinEq.(14)foreverytask,oneafteranother.
TheHessianofthesameunnormalizedlogposterioralsospecifiestheposteriorcovarianceΣ(t):
|      | (cid:16) |     |     |              |      |     |     | (cid:17)−1 |     |
| ---- | -------- | --- | --- | ------------ | ---- | --- | --- | ---------- | --- |
| Σ(t) | ∇2       |     |     | (cid:80) t − | 1 ∇2 |     |     | γ−2I       |     |
= logp(D t |θ)| + logp(D t(cid:48) |θ)| θ(t(cid:48)) + . (16)
|     | θ                    |                    | θ(t) | t (cid:48) =                  | 1 θ                |     |                           |                                      |     |
| --- | -------------------- | ------------------ | ---- | ----------------------------- | ------------------ | --- | ------------------------- | ------------------------------------ | --- |
|     |                      |                    | MAP  |                               |                    |     | MAP                       | (cid:124)(cid:123)(cid:122)(cid:125) |     |
|     | (cid:124)            | (cid:123)(cid:122) |      | (cid:125) (cid:124)           | (cid:123)(cid:122) |     | (cid:125) logpriorHessian |                                      |     |
|     | loglikelihoodHessian |                    |      | previousloglikelihoodHessians |                    |     |                           |                                      |     |
ThissummationoverHessiansistypicallyintractableforneuralnetworkswithlargeparametervectors
θ andhencediagonalorKFACapproximationsareused[2,99,24]. Forthediagonalversion,the
additionofHessiansandlogpriorisexact.FortheKFACversion,wefollowthealternativesuggestion
byRitteretal.[24]andaddupKroneckerfactorswhichisanapproximationtothesumofKronecker
products.However,thisapproximationiswhatunderliesKFACeveninthesupervisedlearningcase
whereweaddupfactorsperdatapointovertheentiredataset.Lastly,weadaptγ duringtraining
oneachtasktbyoptimizingthemarginallikelihoodp(D |D ,...D ),i.e.,bydifferentiatingit
|     |     |     |     |     |     | t 1 | t−1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
withrespecttoγ.ThiscanbedonebycomputingtheeigendecompositionofthesummedKronecker
factors [22] and allows us to 1) adjust the regularization suitably per task and 2) avoid setting a
hyperparametertherebyalleviatingtheneedforvalidationdata.
11Thecodeisavilableathttps://github.com/team-approx-bayes/fromp.
27

Table5:ThememorycomplexitiesofallmethodsinOnotation.Togetabetterideaofwhatthese
complexitiestranslatetoinpractice,wealsoreporttheactualmemoryfootprints(inmegabytes)of
aWideResNet16-4(WRN)onCIFAR-10.Here,M denotesthenumberofmodelparameters,H
denotesthenumberofneuronsinthelastlayer,K denotesthenumberofmodeloutputs,Rdenotes
thenumberofSWAGsnapshots,S denotesthenumberofCSGHMCsamples,andN denotesthe
numberofdeepensemble(DE)members.Mean-fieldvariationalinference(VB)hasacomplexity
of2M asitneedstostoreavariancevectorofsizeM inadditiontothemeanvectorofsizeM.For
the actualmemoryfootprints,we assume R = 40 SWAG snapshots,S = 12 CSGHMC samples,
andN =5ensemblemembers,whicharethehyperparametersrecommendedintheoriginalpapers
(andthereforealsousedinourexperiments).ItcanbeseenthattheproposeddefaultKFAC-last-layer
approximationposesasmallmemoryoverheadofO(H2+K2)ontopoftheMAPestimate.
| METHOD | MEM.COMPLEXITY WRNONCIFAR-10 |       |
| ------ | ---------------------------- | ----- |
| MAP    | M                            | 11MB  |
| LA     | M+H2+K2                      | 12MB  |
| VB     | 2M                           | 22MB  |
| DE     | NM                           | 55MB  |
| CSGHMC | SM                           | 132MB |
RM
| SWAG |     | 440MB |
| ---- | --- | ----- |
C.5 ComparisonofMemoryComplexity
Table5comparesthetheoreticalmemorycomplexityandactualmemoryfootprint(ofaWideResNet
16-4onCIFAR-10)ofthedifferentmethods.
28