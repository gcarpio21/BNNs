| Kerrie    | L. Mengersen |     | (cid:129) Pierre | Pudlo (cid:129) |
| --------- | ------------ | --- | ---------------- | --------------- |
| Christian | P. Robert    |     |                  |                 |
Editors
| Case     | Studies     |      | in     | Applied   |
| -------- | ----------- | ---- | ------ | --------- |
| Bayesian |             | Data |        | Science   |
| CIRM     | Jean-Morlet |      | Chair, | Fall 2018 |

Editors
KerrieL.Mengersen PierrePudlo
MathematicalSciences I2M,CNRS,CentraleMarseille
QueenslandUniversityofTechnology Aix-MarseilleUniversity
Brisbane,QLD,Australia Marseille,France
ChristianP.Robert
CEREMADE
UniversitéParisDauphine
Paris,France
ISSN0075-8434 ISSN1617-9692 (electronic)
LectureNotesinMathematics
ISBN978-3-030-42552-4 ISBN978-3-030-42553-1 (eBook)
https://doi.org/10.1007/978-3-030-42553-1
Mathematics Subject Classification (2020): 62R07, 62F15, 60GXX, 62H30, 62P10, 62M40, 62G05,
60J10
JointlypublishedwithSociétéMathématiquedeFrance(SMF);soldanddistributedtoitsmembersby
theSMF,http://smf.emath.fr;ISBNSMF:978-2-85629-914-2
©TheEditor(s)(ifapplicable)andTheAuthor(s),underexclusivelicencetoSpringerNatureSwitzerland
AG2020
Thisworkissubjecttocopyright.AllrightsaresolelyandexclusivelylicensedbythePublisher,whether
thewhole orpart ofthematerial isconcerned, specifically therights oftranslation, reprinting, reuse
ofillustrations, recitation, broadcasting, reproductiononmicrofilmsorinanyotherphysicalway,and
transmissionorinformationstorageandretrieval,electronicadaptation,computersoftware,orbysimilar
ordissimilarmethodologynowknownorhereafterdeveloped.
Theuseofgeneraldescriptivenames,registerednames,trademarks,servicemarks,etc.inthispublication
doesnotimply,evenintheabsenceofaspecificstatement,thatsuchnamesareexemptfromtherelevant
protectivelawsandregulationsandthereforefreeforgeneraluse.
Thepublisher,theauthors,andtheeditorsaresafetoassumethattheadviceandinformationinthisbook
arebelievedtobetrueandaccurateatthedateofpublication.Neitherthepublishernortheauthorsor
theeditorsgiveawarranty,expressedorimplied,withrespecttothematerialcontainedhereinorforany
errorsoromissionsthatmayhavebeenmade.Thepublisherremainsneutralwithregardtojurisdictional
claimsinpublishedmapsandinstitutionalaffiliations.
ThisSpringerimprintispublishedbytheregisteredcompanySpringerNatureSwitzerlandAG.
Theregisteredcompanyaddressis:Gewerbestrasse11,6330Cham,Switzerland

Chapter 3
Bayesian Neural Networks:
An Introduction and Survey
EthanGoanandClintonFookes
Abstract NeuralNetworks(NNs) have providedstate-of-the-artresults for many
challengingmachinelearningtaskssuchasdetection,regressionandclassification
acrossthedomainsofcomputervision,speechrecognitionandnaturallanguagepro-
cessing.Despitetheirsuccess,theyareoftenimplementedinafrequentistscheme,
meaningtheyareunabletoreasonaboutuncertaintyintheirpredictions.Thisarticle
introducesBayesian NeuralNetworks(BNNs) andthe seminalresearchregarding
theirimplementation.Differentapproximateinferencemethodsarecompared,and
usedtohighlightwherefutureresearchcanimproveoncurrentmethods.
3.1 Introduction
Biomimicry has long served as a basis for technologicaldevelopments.Scientists
and engineers have repeatedly used knowledge of the physical world to emulate
nature’selegantsolutionstocomplexproblemswhichhaveevolvedoverbillionsof
years.Animportantexampleofbiomimicryin statisticsandmachinelearninghas
beenthedevelopmentoftheperceptron[1],whichproposesamathematicalmodel
basedonthephysiologyofaneuron.Themachinelearningcommunityhasusedthis
concept1todevelopstatisticalmodelsofhighlyinterconnectedarraysofneuronsto
createNeuralNetworks(NNs).
ThoughtheconceptofNNshasbeenknownformanydecades,itisonlyrecently
thatapplicationsofthesenetworkhaveseensuchprominence.Thelullinresearch
and development for NNs was largely due to three key factors: lack of sufficient
algorithms to train these networks, the large amount of data required to train
1Whilealsorelaxingmanyoftheconstraintsimposedbyaphysicalmodelofanaturalneuron[2].
E.Goan((cid:2))·C.Fookes
QueenslandUniversityofTechnology,Brisbane,QLD,Australia
e-mail:ej.goan@qut.edu.au
©TheEditor(s)(ifapplicable)andTheAuthor(s),underexclusive 45
licencetoSpringerNatureSwitzerlandAG2020
K.L.Mengersenetal.(eds.),CaseStudiesinAppliedBayesianDataScience,
LectureNotesinMathematics2259,https://doi.org/10.1007/978-3-030-42553-1_3

46 E.GoanandC.Fookes
complex networks and the large amount of computing resources required during
the trainingprocess.In 1986,Rumelhartetal. [3] introducedthe backpropagation
algorithmtoaddresstheproblemofefficienttrainingforthesenetworks.Thoughan
efficientmeansoftrainingwasavailable,considerablecomputeresourceswasstill
requiredfortheeverincreasingsizeofnewnetworks.Thisproblemwasaddressed
in[4–6]whereitwasshownthatgeneralpurposeGPUscouldbeusedtoefficiently
performmanyoftheoperationsrequiredfortraining.Asdigitalhardwarecontinued
toadvance,thenumberofsensorsabletocaptureandstorerealworlddataincreased.
With efficient training methods, improvedcomputationalresources and large data
sets,trainingofcomplexNNsbecametrulyfeasible.
Inthevastmajorityofcases,NNsareusedwithinafrequentistperspective;using
availabledata,auserdefinesanetworkarchitectureandcostfunction,whichisthen
optimised to allow us to gain pointestimate predictions.Problemsarise from this
interpretationofNNs.Increasingthenumberofparameters(oftencalledweightsin
machinelearningliterature),orthedepthofthemodelincreasesthecapacityofthe
network,allowingittorepresentfunctionswithgreaternon-linearities.Thisincrease
incapacityallowsformorecomplextaskstobeaddressedwithNNs,thoughwhen
frequentistmethodologiesareapplied,leavesthemhighlypronetooverfittingtothe
trainingdata.Theuseoflargedatasetsandregularisationmethodssuchasfindinga
MAPestimatecanlimitthecomplexityoffunctionslearntbythenetworksandaid
inavoidingoverfitting.
Neural Networks have provided state-of-the-art results for numerous machine
learning and Artificial intelligence (AI) applications, such as image classification
[6–8], object detection [9–11] and speech recognition [12–15]. Other networks
such as the AlphaGo model developed by DeepMind [16] have emphasised the
potentialof NNs for developingAI systems, garneringa wide audienceinterested
in the developmentof these networks. As the performanceof NNs has continued
to increase, the interest in their development and adoption by certain industries
becomes more prominent. NNs are currently used in manufacturing [17], asset
management[18]andhumaninteractiontechnologies[19,20].
SincethedeploymentofNNsinindustry,therehavebeenanumberofincidents
wherefailingsin these systems has led to modelsacting unethicallyand unsafely.
This includes models demonstrating considerable gender and racial bias against
marginalisedgroups[21–23]ortomoreextremecasesresultinginlossoflife[24,
25]. NNs are a statistical black-box models, meaning that the decision process is
not based on a well-defined and intuitive protocol. Instead decisions are made in
an uninterpretablemanner,with hopes that the reasonable decisions will be made
basedonpreviousevidenceprovidedintrainingdata.2Assuch,theimplementation
ofthesesystemsinsocialandsafetycriticalenvironmentsraisesconsiderableethical
concerns.TheEuropeanUnionreleasedanewregulation3 whicheffectivelystates
2Due to this black-box nature, the performance of these models is justified entirely through
empiricalmeans.
3Thisregulationcameintoeffectonthe25thofMay,2018acrosstheEU[26].

3 BayesianNeuralNetworks:AnIntroductionandSurvey 47
(a) (b)
Fig.3.1 Comparisonofneuralnetworktotraditionalprobabilisticmethodsforaregressiontask,
withnotrainingdatainthepurpleregion. (a)Regression outputusinganeural network with2
hiddenlayers;(b)RegressionusingaGaussianProcessframework,withgreybarrepresenting±2
std.fromexpectedvalue
thatusershavea“righttoanexplanation”regardingdecisionsmadebyAIsystems
[26,27]. Withoutclear understandingof their operationor principledmethodsfor
theirdesign,expertsfromotherdomainsremainapprehensiveabouttheadoptionof
currenttechnology[28–30].Theselimitationshavemotivatedresearcheffortsinto
thefieldofExplainableAI[31].
AdequateengineeringofNNsrequiresasoundunderstandingoftheircapabilities
and limitations; to identify their shortcomings prior to deployment as apposed
to the current practice of investigating these limitations in the wake of these
tragedies. With NNs being a statistical black-box, interpretation and explanation
of the decision making process eludes current theory. This lack of interpretation
and over-confident estimates provided by the frequentist perspective of common
NNs makes them unsuitable for high risk domains such as medical diagnostics
and autonomous vehicles. Bayesian statistics offers natural way to reason about
uncertainty in predictions, and can provide insight into how these decisions are
made.
Figure 3.1 compares Bayesian methods for performing regression with that of
a simple neural network, and illustrates the importance of measuring uncertainty.
While both methods perform well within the bounds of the training data, where
extrapolation is required, the probabilistic method provides a full distribution
of the function output as opposed to the point estimates provided by the NN.
The distribution over outputs provided by probabilistic methods allows for the
development of trustworthy models, in that they can identify uncertainty in a
prediction.GiventhatNNsarethemostpromisingmodelforgeneratingAIsystems,
itisimportantthatwecansimilarlytrusttheirpredictions.
A Bayesian perspective allows us to address many of the challenges currently
facedwithinNNs.Todothis,adistributionisplacedoverthenetworkparameters,
and the resultingnetwork is then termeda Bayesian NeuralNetwork (BNN). The

48 E.GoanandC.Fookes
goal of a BNN is to have a model of high capacity that exhibits the important
theoretical benefits of Bayesian analysis. Recent research has investigated how
Bayesian approximations can be applied to NNs in practice. The challenge with
thesemethodsisdeployingmodelsthatprovideaccuratepredictionswithinreason-
ablecomputationconstraints.4
ThisdocumentaimstoprovideanaccessibleintroductiontoBNNs,accompanied
byasurveyofseminalworksinthefieldandexperimentstomotivatediscussioninto
thecapabilitiesandlimitsofcurrentmethods.Asurveyofallresearchitemsacross
the Bayesian and machine learning literature related to BNNs could fill multiple
text books. As a result, items included in this survey only intend to inform the
reader on the overarching narrative that has motivated their research. Similarly,
derivations of many of they key results have been omitted, with the final result
being listed accompaniedby referenceto the originalsource. Readers inspired by
this exciting research area are encouraged to consult prior surveys: [32] which
surveystheearlydevelopmentsinBNNs,[33]whichdiscussesthespecificsofafull
Bayesian treatment for NNs, and [34] which surveys applications of approximate
Bayesianinferencetomodernnetworkarchitectures.
Thisdocumentshouldbesuitableforallinthestatisticsfield,thoughtheprimary
audienceofinterestarethosemorefamiliarwithmachinelearningconcepts.Despite
seminal references for new machine learning scholars almost equivalently being
Bayesian texts [2, 35], in practice there has been a divergence between much of
themodernmachinelearningandBayesianstatisticsresearch.Itishopedthatthis
surveywillhelphighlightsimilaritiesbetweensomemodernresearchinBNNsand
statistics,toemphasistheimportanceofaprobabilisticperspectivewithinmachine
learningandtoencouragefuturecollaboration/unisonbetweenthemachinelearning
andstatisticsfields.
3.2 Literature Survey
3.2.1 Neural Networks
Beforediscussing a Bayesian perspectiveofNNs, it is importantto briefly survey
the fundamentals of neural computation and to define the notation to be used
throughout the chapter. This survey will focus on the primary network structure
of interest, the Multi-Layer Perceptron (MLP) network. The MLP serves as the
basis for NNs, with modern architectures such as convolutional networks having
anequivalentMLPrepresentation.Figure3.2illustratesasimpleMLPwithasingle
hiddenlayersuitableforregressionorclassification.Forthisnetworkwithaninput
4Theterm“reasonable”largelydependsonthecontext.Manyneuralnetworksarecurrentlytrained
usingsomeofthelargestcomputingfacilitiesavailable,containingthousandsofGPUdevices.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 49
Fig.3.2 ExampleofaNN Input Hidden Output
architecturewithasingle Layer Layer Layer
hiddenlayerforeitherbinary x φ(x) f(x)
classificationor1-D
regression.Eachnode
representsaneuronorastate
wherethesummationand
activationofinputstatesis x1
performed.Arrowsarethe
parameters(weights)
indicatingthestrengthof
connectionbetweenneurons
x2 Output
x3
xofdimensionN ,theoutputofthefnetworkcanbemodelledas,
1
(cid:2)N1
φ = a(x w1), (3.1)
j i ij
i=1
(cid:2)N2
f = g(φ w2 ). (3.2)
k j jk
j=1
The parameters w represent the weighted connection between neurons from sub-
sequent layers, and the superscripts denoting the layer number. Equation (3.1)
representsthe outputof the hiddenlayer,whichwill be ofdimensionN . Thekth
2
outputofthenetworkisthenasummationovertheN outputsfromthepriorhidden
2
layer.Thismodellingschemecanbeexpandedtoincludemanyhiddenlayers,with
theinputofeachlayerbeingtheoutputofthelayerimmediatelyprior.Abiasvalue
isoftenaddedduringeachlayer,thoughisomittedthroughoutthischapterinfavour
ofsimplicity.
Equation(3.1)referstothestateofeachneuron(ornode)inthehiddenlayer.This
isexpressedasanaffinetransformfollowedbyanon-linearelementwisetransform
φ(·), which is often called an activation. For the original perceptron, activation
functionused was the sign(·) function,thoughthe use of this functionhas ceased
due to it’s derivative being equal to zero.5 More favourable activation functions
such as the Sigmoid, Hyperbolic Tangent (TanH), Rectified Linear Unit (ReLU)
and Leaky-ReLU have since replaced this the sign function [36, 37]. Figure 3.3
illustrates these functionsalong with their correspondingderivatives. When using
5Whenthederivativeisdefined,asisapiece-wisenon-differentiablefunctionattheorigin.

50 E.GoanandC.Fookes
(a) (b)
(c) (d)
Fig.3.3 ExamplesofcommonlyusedactivationfunctionsinNNs.Theoutputforeachactivation
isshowninblueandthenumericalderivativeofeachfunctionisshowninred.Thesefunctionsare
(a)Sigmoid;(b)TanH;(c)ReLU;(d)Leaky-ReLU.Notethechangeinscaleforthey-axis
theSigmoidfunction,expression(3.1)isequivalenttologisticregression,meaning
that the output of the network becomes the sum of multiple logistic regression
models.
Foraregressionmodel,thefunctionappliedtotheoutputg(·)willbetheidentity
function,6andforbinaryclassificationwillbeaSigmoid.
Equations (3.1) and (3.2) can be efficiently implemented using matrix repre-
sentations, and is often representedas such in machine learning literature. This is
achieved by stacking the input vector in our data set as a column in X. Forward
propagationcanthenbeperformedas,
(cid:2)=a(XTW1), (3.3)
F=g((cid:2)W2). (3.4)
6Meaningnoactivationisusedontheoutputlayer,g(x)=x.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 51
Whilst this matrix notation is more concise, the choice to use the summation
notation to describe the network here is deliberate. It is hoped that with the
summationnotation,relationstokernelandstatisticaltheorydiscussedlaterinthis
chapterbecomesclearer.
In the frequentist setting of NN learning, a MLE or MAP estimate is found
through the minimisation of a non-convex cost function J(x,y) w.r.t. network
weights.Minimisationofthiscost-functionisperformedthroughbackpropagation,
wheretheoutputofthemodeliscomputedforthecurrentparametersettings,partial
derivativesw.r.tparametersarefoundandthenusedtoupdateeachparameter,
∂J(x,y)
w t+i =w t −α . (3.5)
∂w
t
Equation (3.5) illustrates how backpropagationupdatesmodel parameters, with α
representingthelearningrateandthesubscriptsindicatetheiterationinthetraining
procedure. Partial derivatives for individual parameters at different layers in the
networkisfoundthroughapplicationofthechainrule.Thisleadstothepreference
of discontinuous non-linearities such as the ReLU for deep NNs, as the larger
gradientoftheReLUassistsinpreventingvanishinggradientsofearlylayersduring
training.
3.2.2 Bayesian Neural Networks
In the frequentist setting presented above, the model weights are not treated as
random variables; weights are assumed to have a true value that is just unknown
and the data we have seen is treated as a random variable. This may seem
counterintuitive for what we want to achieve. We would like to learn what our
unknownmodelweightsarebasedoftheinformationwehaveathand.Forstatistical
modellingthe informationavailable to us comesin the form ofour acquireddata.
Sincewedonotknowthevalueforourweights,itseemsnaturaltotreatthemasa
randomvariable.The Bayesian view ofstatistics uses thisapproach;unknown(or
latent)parametersaretreatedasrandomvariablesandwewanttolearnadistribution
oftheseparametersconditionalonthewhatwecanobserveinthetrainingdata.
During the “learning” process of BNNs, unknown model weights are inferred
basedonwhatwedoknoworwhatwecanobserve.Thisistheproblemofinverse
probability, and is solved through the use of Bayes Theorem. The weights in our
model ω are hidden or latent variables; we cannot immediately observe their true
distribution.BayesTheoremallowsustorepresentadistributionovertheseweights
in terms of probabilities we can observe, resulting in the distribution of model

52 E.GoanandC.Fookes
p(ω|D),7
parameters conditional on the data we have seen which we call the
posteriordistribution.
Before training,we can observethe joint distributionbetween our weights and
p(ω,D).
our data This joint distribution is defined by our prior beliefs over our
latentvariablesp(ω)andourchoiceofmodel/likelihoodp(D|ω),
| p(ω,D)=p(ω)p(D|ω). |     |     | (3.6) |
| ------------------ | --- | --- | ----- |
Ourchoiceofnetworkarchitectureandlossfunctionisusedtodefinethelikelihood
terminEq.(3.6).Forexample,fora1-Dhomoscedasticregressionproblemwitha
mean squared error loss and a knownnoise variance,the likelihood is a Gaussian
distributionwiththemeanvaluespecifiedbytheoutputofthenetwork,
|          | (cid:3)  | (cid:4) |     |
| -------- | -------- | ------- | --- |
| p(D|ω)=N | fω(D),σ2 |         |     |
.
D
Under this modellingscheme, it is typically assumed that all samples from are
i.i.d.,meaningthatthelikelihoodcanthenbewrittenasaproductofthecontribution
fromtheN individualtermsinthedataset,
|         | (cid:5)N (cid:3) | (cid:4) |       |
| ------- | ---------------- | ------- | ----- |
| p(D|ω)= | N fω(x ),σ2      | .       | (3.7) |
i
i=1
Our prior distribution should be specified to incorporate our belief as to how the
weights should be distributed, prior to seeing any data. Due to the black-box
nature of NNs, specifying a meaningful prior is challenging. In many practical
NNstrainedunderthefrequentistscheme,theweightsofthetrainednetworkhave
a low magnitude, and are roughly centred around zero. Following this empirical
observation,wemayuseazeromeanGaussianwithasmallvarianceforourprior,
oraspike-slabpriorcentredatzerotoencouragesparsityinourmodel.
With the prior and likelihood specified, Bayes theorem is then applied to yield
theposteriordistributionoverthemodelweights,
| p(ω)p(D|ω)      |     | p(ω)p(D|ω) |       |
| --------------- | --- | ---------- | ----- |
| π(ω|D)= (cid:6) | =   |            |       |
|                 |     | .          | (3.8) |
| p(ω)p(D|ω)dω    |     | p(D)       |       |
Thedenominatorintheposteriordistributioniscalledthemarginallikelihood,orthe
evidence.This quantity is a constant with respect to the unknownmodel weights,
andnormalisestheposteriortoensureitisavaliddistribution.
7Disusedheretodenotethesetoftrainingdata(x,y).

3 BayesianNeuralNetworks:AnIntroductionandSurvey 53
From this posteriordistribution, we can performpredictionsof any quantityof
interest.Predictionsare intheformofanexpectationwith respecttotheposterior
distribution,
(cid:7)
E [f]= f(ω)π(ω|D)dω. (3.9)
π
Allpredictivequantitiesofinterestwillbeanexpectationofthisform.Whetheritbe
apredictivemean,varianceorinterval,thepredictivequantitywillbeanexpectation
over the posterior. The only change will be in the function f(ω) with which the
expectationisappliedto.Predictioncanthenbeviewedasanaverageofthefunction
f weightedbytheposteriorπ(ω).
We see that the Bayesian inference process revolves around marginalisation
(integration) over our unknown model weights. By using this marginalisation
approach,weareabletolearnaboutthegenerativeprocessofamodel,asopposedto
anoptimisationschemeusedinthefrequentistsetting.Withaccesstothisgenerative
model,ourpredictionsarerepresentedintheformofvalidconditionalprobabilities.
In this description, it was assumed that many parameters such as the noise
variance σ or any prior parameters were known. This is rarely the case, and as
such we need to perform inference for these unknown variables. The Bayesian
framework allows us to perform inference over these variables similarly to how
weperforminferenceoverourweights;wetreattheseadditionalvariablesaslatent
variables, assign a prior distribution (or sometimes called a hyper-prior)and then
marginaliseoverthem to find ourposterior.For more of a descriptionof how this
canbeperformedforBNNs,pleasereferto[33,38].
For many models of interest, computation of the posterior (Eq. (3.8)) remains
intractable. This is largely due to the computationof the marginallikelihood. For
non-conjugatemodels or those that are non-linear in the latent variables (such as
NNs),thisquantitycanbeanalyticallyintractable.Forhighdimensionalmodels,a
quadratureapproximationofthisintegralcanbecomecomputationallyintractable.
Asaresult,approximationsfortheposteriormustbemade.Thefollowingsections
detailhowapproximateBayesianinferencecanbeachievedinBNNs.
3.2.3 OriginofBayesian Neural Networks
From this survey and those conducted prior [70], the first instance of what could
beconsideredaBNNwasdevelopedin[39].Thispaperemphasiseskeystatistical
propertiesof NNs bydevelopinga statistical interpretationof loss functionsused.
It was shown that minimisation of a squared error term is equivalent to finding
the Maximum Likelihood Estimate (MLE) of a Gaussian. More importantly, it
was shown that by specifying a prior over the network weights, Bayes Theorem
can be used to obtain an appropriate posterior. Whilst this work provides key
insights into the Bayesian perspective of NNs, no means for finding the marginal

54 E.GoanandC.Fookes
likelihood(evidence)issupplied,meaningthatnopracticalmeansforinferenceis
suggested.DenkerandLeCun[40]extendonthiswork,offeringapracticalmeans
for performing approximate inference using the Laplace approximation, though
minimalexperimentalresultsareprovided.
ANNisagenericfunctionapproximator.Itiswellknownthatasthelimitofthe
number of parameters approaches infinity in a single hidden layer network, any
arbitrary function can be represented [41–43]. This means that for the practical
case, our finite training data set can be well approximated by a single layer NN
as long as there are sufficient trainable parameters in the model. Similar to high-
degree polynomial regression, although we can represent any function and even
exactly match the training data in certain cases, as the number of parametersin a
NNincreasesorthedegreeofthepolynomialusedincreases,themodelcomplexity
increases leading to issues of overfitting. This leads to a fundamental challenge
foundinNNdesign;howcomplexshouldImakemymodel?
Building on the work of Gull and Skilling [44], MacKay demonstrates how a
Bayesian framework naturally lends itself to handle the task of model design and
comparisonofgenericstatisticalmodels[45].Inthiswork,twolevelsofinference
aredescribed:inferenceforfittingamodelandinferenceforassessingthesuitability
of a model.The first levelofinferenceis the typicalapplicationofBayes rule for
updatingmodelparameters,
P(D|ω,H )P(ω|H )
P(ω|D,H )= i i , (3.10)
i P(D|H )
i
whereω isthesetofparametersinthegenericstatisticalmodel,D isourdataand
H representsthei’thmodelusedforthislevelofinference.8Thisisthendescribed
i
as,
Likelihood×Prior
Posterior= .
Evidence
ItisimportanttonotethatthenormalisingconstantinEq.(3.10)isreferredtoasthe
evidenceforthespecificmodelofinterestH . Evaluationoftheposteriorremains
i
intractable for most models of interest, so approximations must be made. In this
work,theLaplaceapproximationisused.
Though computation of the posterior over parameters is required, the key aim
of this work is to demonstrate methods of assessing the posterior over the model
hypothesisH .Theposteriorovermodeldesignisrepresentedas,
i
P(H |D)∝P(D|H )P(H ), (3.11)
i i i
8Hisusedtorefertothemodel“hypothesis”.

| 3 BayesianNeuralNetworks:AnIntroductionandSurvey |     |     |     |     | 55  |
| ------------------------------------------------ | --- | --- | --- | --- | --- |
whichtranslatesto,
ModelPosterior∝Evidence×ModelPrior.
The data dependent term in Eq. (3.11) is the evidence for the model. Despite the
promisinginterpretationoftheposteriornormalisationconstant,asdescribedearlier,
evaluationof thisdistributionis intractablefor mostBNNs. Assuming a Gaussian
distribution,theLaplaceapproximationoftheevidencecanbefoundas,
(cid:7)
| P(D|H | )= P(D|ω,H | )P(ω|H   | )dω     |         |        |
| ----- | ---------- | -------- | ------- | ------- | ------ |
|       | i          | i        | i       |         | (3.12) |
|       |            | (cid:8)  | (cid:9) |         |        |
|       | ≈P(D|ω     | ,H ) P(ω | |H )Δω  |         | (3.13) |
|       |            | MAP i    | MAP i   |         |        |
|       |            | (cid:8)  |         | (cid:9) |        |
k −1
|     | =P(D|ω | ,H ) P(ω | |H )(2π) 2det | 2A  | (3.14) |
| --- | ------ | -------- | ------------- | --- | ------ |
|     |        | MAP i    | MAP i         |     |        |
=BestLikelihoodFit×OccamFactor.
Thiscan be interpretedas a single Riemannapproximationto the modelevidence
with the best likelihood fit representing the peak of the evidence, and the Occam
factor is the width that is characterised by the curvature around the peak of the
Gaussian. The Occam factor can be interpreted as the ratio of the width of the
| posteriorΔωandtherangeofthepriorΔω |     |     | forthegivenmodelH | ,   |     |
| ---------------------------------- | --- | --- | ----------------- | --- | --- |
|                                    |     |     | 0                 | i   |     |
Δω
OccamFactor=
|     |     |     | ,   |     | (3.15) |
| --- | --- | --- | --- | --- | ------ |
Δω
0
meaningthatthe Occamfactor isthe ratio ofchangein plausibleparameterspace
from the prior to the posterior. Figure 3.4 demonstrates this concept graphically.
Evidence
P(D|H 1)
|     |     |     | P(D|H 2) |     |     |
| --- | --- | --- | -------- | --- | --- |
D
Fig.3.4 Graphicalillustrationofhowtheevidenceplaysaroleininvestigatingdifferent model
hypotheses. ThesimplemodelH isabletopredictasmallrangeofdatawithgreaterstrength,
1
whilethemorecomplexmodelH isabletorepresentalargerrangeofdata,thoughwithlower
2
probability.Adaptedfrom[45,46]

56 E.GoanandC.Fookes
With this representation, a complex model able to represent a large range of data
willhaveawiderevidence,thushavingalargerOccamfactor.Asimplemodelwill
havealowercapacitytocaptureacomplexgenerativeprocess,butasmallerrangeof
datawillbeabletobemodelledwithgreatercertainty,resultinginalowerOccam
Factor. This results in a natural regularisation for the complexity of a model. An
unnecessarilycomplexmodelwilltypicallyresultinawideposterior,resultingina
largeOccamfactorandlowevidenceforthegivenmodel.Similarly,awideorless
informativepriorwillresultina reducedOccamfactor,providingfurtherintuition
intotheBayesiansettingofregularisation.
Usingthisevidenceframeworkrequirescomputationofthemarginallikelihood,
which is an expensive (and the key challenge) within Bayesian modelling. Given
the large investment required to approximate the marginal likelihood, it may
be infeasible to compare many different architectures. Despite this, the use of
the evidence framework can used to assess solutions for BNNs. For most NN
architectures of interest, the objective function is non-convex with many local
minima.Eachlocalminimacanberegardedasapossiblesolutionfortheinference
problem. MacKay uses this as motivation to compare the solutions from each
local minimum using the corresponding evidence function [47]. This allows for
assessmentofmodelcomplexityateachsolutionwithoutprohibitivecomputational
requirements.
3.2.3.1 EarlyVariationalInferenceforBNNs
Themachinelearningcommunityhascontinuouslyexcelledatoptimisationbased
problems. While many ML models, such as SupportVector Machines and Linear
Gaussian Models result in a convex objective function, NNs have a highly non-
convex objective function with many local minima. A difficult to locate global
minimum motivates the use of gradient based optimisation schemes such as
backpropagation[3].ThistypeofoptimisationcanbeviewedinaBayesiancontext
throughthelensofVariationalInference(VI).
VI is an approximate inference method that frames marginalisation required
during Bayesian inference as an optimisation problem [48–50]. This is achieved
by assumingthe form of the posteriordistributionand performingoptimisationto
findtheassumeddensitythatclosesttothetrueposterior.Thisassumptionsimplifies
computationandprovidessomeleveloftractability.
The assumed posterior distribution qθ(ω) is a suitable density over the set of
parametersω,thatisrestrictedtoacertainfamilyofdistributionsparameterisedby
θ. The parameters for this variational distribution are then adjusted to reduce the
dissimilarity between the variational distribution and the true posterior p(ω|D).9
9ThemodelhypothesisH i usedpreviouslywillbeomittedforfurtherexpressions,aslittleofthe
remainingkeyresearchitemsdealwithmodelcomparisonandsimplyassumeasinglearchitecture
andsolution.

| 3 BayesianNeuralNetworks:AnIntroductionandSurvey |     |     |     | 57  |
| ------------------------------------------------ | --- | --- | --- | --- |
ThemeanstomeasuresimilarityforVIisoftentheforwardKL-Divergencebetween
thevariationalandtruedistribution,
(cid:7)
(cid:10) (cid:11)
qθ(ω)
| KL qθ(ω)||p(ω|D) |     | = qθ(ω)log | dω. | (3.16) |
| ---------------- | --- | ---------- | --- | ------ |
p(ω|D)
For VI, Eq. (3.16) serves as the objective function we wish to minimise w.r.t
variationalparametersθ.Thiscanbeexpandedoutas,
| (cid:10)         | (cid:11) (cid:12) | qθ(ω)             | (cid:13)            |        |
| ---------------- | ----------------- | ----------------- | ------------------- | ------ |
| KL qθ(ω)||p(ω|D) | =E                | log −logp(D|ω)    | +logp(D)            | (3.17) |
|                  | q                 | p(ω)              |                     |        |
|                  |                   | (cid:10) (cid:11) |                     |        |
|                  | =KL               | qθ(ω)||p(ω) −E    | [logp(D|ω)]+logp(D) |        |
q
(3.18)
|     | = −F[qθ | ]+logp(D), |     |     |
| --- | ------- | ---------- | --- | --- |
(3.19)
|     | (cid:10) | (cid:11) |     |     |
| --- | -------- | -------- | --- | --- |
whereF[qθ ] =−KL qθ(ω)||p(ω) +E [logp(D|ω)].Thecombinationofterms
q
into F[q] is to separate the tractable terms from the intractable log marginal
likelihood.Wecannowoptimisethisfunctionusingbackpropagation,andsincethe
logmarginallikelihooddoesnotdependonvariationalparametersθ,it’sderivative
evaluatestozero.Thisleavesonlytermofcontainingvariationalparameters,which
isF[qθ ].
ThisnotationusedinEq.(3.19),particularlythechoicetoincludethenegativeof
F[qθ ]isdeliberatetohighlightadifferentbutequivalentderivationtotheidentical
result,andtoremainconsistentwithexistingliterature.Thisresultcanbeobtained
by instead of minimising the KL-Divergence between the true and approximate
distribution,butbyapproximatingtheintractablelogmarginallikelihood.Through
applicationofJensen’sinequality,wecanthenfindthatF[qθ ]formsalowerbound
on the logarithm of the marginal likelihood [48, 51]. This can be seen by re-
arranging Eq. (3.19) and noting that the KL divergence is strictly ≥ 0 and only
equals zero when the two distributions are equal. The logarithm of the marginal
likelihood is equal to the sum of the KL divergence between the approximate
and true posterior and F[qθ ]. By minimising the KL divergence between the
|     |     | F[qθ ] |     |     |
| --- | --- | ------ | --- | --- |
approximate and true posterior, the closer will be to the logarithm of the
marginallikelihood.Forthisreason,F[qθ ]iscommonlyreferredtoastheEvidence
LowerBound(ELBO).Figure3.5illustratesthisgraphically.
The first applicationof VI to BNNs was byHintonand VanCamp [53], where
theytriedtoaddresstheproblemofoverfittinginNNs.Theyarguedthatbyusinga
probabilistic perspective of model weights, the amount of information they could
contain would be reduced and would simplify the network. Formulation of this
problem was through an information theoretic basis, particularly the Minimum
DescriptiveLength(MDL)principle,thoughitsapplicationresultsinaframework
equivalentto VI. As is common in VI, the mean-field approachwas used. Mean-
Field Variational Bayes (MFVB) assumes a posterior distribution that factorises

58 E.GoanandC.Fookes
(cid:2) (cid:3)
KL q θ(ω)||p(ω|D)
logp(D)
F[q θ]
Fig.3.5 Graphical illustration of how the minimisation of the KL divergence between the
approximateandtrueposteriormaximisesthelowerboundontheevidence.AstheKLDivergence
betweenourapproximate andtrueposterior isminimised,theELBOF[qθ ]tightenstothelog-
evidence.ThereforemaximisingtheELBOisequivalenttominimisingtheKLdivergencebetween
theapproximateandtrueposterior.Adaptedfrom[52]
over parameters of interest. For the work in [53], the posterior distribution over
modelweightswasassumedtobeafactorisationofindependentGaussians,
(cid:5)P
qθ(ω)= N(w
i
|μ
i
,σ
i
2), (3.20)
i=1
whereP isthenumberofweightsinthenetwork.Foraregressionnetworkwitha
singlehiddenlayer,ananalyticsolutionforthisposteriorisavailable.Theabilityto
achieveananalyticsolutiontotheapproximationisandesirableproperty,asanalytic
solutionssignificantlyreducethetimetoperforminference.
There are a few issues with this work, though one of the most prominent
issues is the assumption of a posterior that factorises over individual network
weights. It is well known that strong correlation between parameters in a NN is
present. A factorised distribution simplifies computation by sacrificing the rich
correlationinformationbetweenparameters.MacKayhighlightedthislimitationin
anearlysurveyofBNNs[32]andoffersinsightintohowapreprocessingstageof
inputstohiddenlayerscouldallowformorecomprehensiveapproximateposterior
distributions.
Barber and Bishop [52] again highlight this limitation, and offer a VI based
approach that extends on the work in [53] to allow for full correlation between
theparameterstobecapturedbyusinga fullrankGaussianfortheapproximating
posterior.For a single hiddenlayer regressionnetwork utilising a Sigmoid activa-
tion,analyticexpressionsforevaluatingtheELBO is provided.10 Thisis achieved
byreplacingtheSigmoidwiththeappropriatelyscalederrorfunction.
10Numerical methods are required to evaluate certain terms in the analytic expression for the
ELBO.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 59
Anissuewiththismodellingschemeistheincreasednumberofparameters.For
a full covariance model, the number of parameters scales quadratically with the
number of weights in the network. To rectify this, Barber and Bishop propose a
restrictedformforthecovarianceoftenusedinfactoranalysis,suchthat,
(cid:2)s
C=diag(d2,...,d2)+ s sT, (3.21)
1 n i i
i=1
wherethediagoperatorcreatesadiagonalmatrixfromthevectordofsizen,where
n is the number of weights in the model. This form then scales linearly with the
numberofhiddenunitsinthenetwork.
These bodies of work provide important insight into how the prominentback-
propagationmethodcanbeappliedtochallengingBayesianproblems.Thisallows
for properties of the two areas of research to be merged and offer the benefits
nominallyseeninisolation.Complexregressiontasksforlargebodiesofdatasets
couldnowbehandledinaprobabilisticsenseusingNNs.
Despite the insight offered by these methods, there are limitations to these
methods. Both the work of Hinton and Van Camp and Barber and Bishop focus
on development of a closed form representation of the networks.11 This analytic
tractability imposes many restrictions on the networks. As discussed previously,
[53]assumeafactorisedposterioroverindividualweightswhichisunabletocapture
anycorrelationinparameters.Covariancestructureiscapturedin [52], thoughthe
authorslimittheiranalysistotheuseofaSigmoidactivationfunction(whichiswell
approximatedbytheerrorfunction),whichisseldomusedinmodernnetworksdue
to the lowmagnitudein the gradient.12 A keylimitation commonto bothof these
approachesistherestrictionofasinglehiddenlayernetwork.
As stated previously, a NN can approximate any function arbitrarily well by
addingadditionalhiddenunits.Formodernnetworks,empiricalresultshaveshown
that similarly complex functions can be represented with fewer hidden units by
increasing the number of hidden layers in the network. This has lead to the term
“deeplearning”,wheredepthreferstothenumberofhiddenlayers.Thereductionin
numberofweightvariablesisespeciallyimportantforwhentryingtoapproximate
the full covariance structure between layers. For example, correlation between
hiddenunitswithinasinglelayermaybecaptured,whileassumingthatparameters
between the different layers are independent. An assumption such as this can
significantly reduce the number of correlation parameters. With modern networks
having hundreds of millions of weights across many layers (with these networks
onlybeingabletoofferpointestimates),theneedtodeveloppracticalprobabilistic
interpretationsbeyondasinglelayerisessential.
11Althoughtherearealargenumberofbenefitstosuchanapproach,asillustratedearlier.
12Analyticresultsmaybeachievableusingotheractivationfunctions,suchasTanH,whichsuffer
lessfromsuchanissue.

60 E.GoanandC.Fookes
3.2.3.2 HybridMonteCarloforBNNs
It is worthwhile at this point to reflect on the actual quantities of interest. So far
the emphasis has been placed on finding good approximations for the posterior,
though the accurate representation of the posterior is usually not the end design
requirement.Themainquantitiesof interestare predictivemomentsandintervals.
We want to make good predictions accompanied by confidence information. The
reason we emphasise computation of the posterior is that predictive moments
and intervals are all computed as expectations of the posterior π(ω|D).13 This
expectationislistedinEq.(3.9),andisrepeatedhereforconvenience,
(cid:7)
E [f]= f(ω)π(ω|D)dω.
π
This is why computationof the posterior is emphasised;accurate predictionsrely
onaccurateapproximationsoftheintractableposterior.
The previous methods employed optimisation based schemes such as VI or
Laplace approximations of the posterior. In doing so, strong assumptions and
restrictions on the form of posterior are enforced. The restrictions placed are
oftencreditedwithinaccuraciesinducedinpredictions,thoughthisisnottheonly
limitation.
As highlighted by Betancourt [54] and Betancourt et al. [55], the expectation
computedforpredictivequantitiesnotjustaprobabilitymass,ittheproductofthe
probability mass and a volume. The probability mass is our posterior distribution
π(ω|D), and the volume dω over which we are integrating. It is likely that for
all modelsof interest, the contributionof the expectationfrom this productof the
densityandvolumewillnotbeatthemaximumforthemass.Thereforeoptimisation
based schemes which consider only the mass can deliver inaccurate predictive
quantities. To make accurate predictions with finite computational resources, we
need to evaluate this expectationnot just when the mass is greatest, but when the
productofthemassandvolumeislargest.Themostpromisingwaytoachievethis
iswithMarkovChainMonteCarlo(MCMC).
MCMC algorithms remains at the forefront of Bayesian research and applied
statistics.14 MCMC is a general approach for sampling from arbitrary and
intractable distributions. The ability to sample from a distribution enables the
useofMonteCarlointegrationforprediction,
(cid:7)
(cid:2)N
1
E [f]= f(ω)π(ω|D)dω≈ f(ω ), (3.22)
π i
N
i=1
13Notethatπisusedtorepresentthetrueposteriordistributionhere,asapposetoqusedpreviously
todenoteanapproximationoftheposterior.
14MCMCisregardedasoneofthemostinfluentialalgorithmsofthetwenty-firstcentury[56].

3 BayesianNeuralNetworks:AnIntroductionandSurvey 61
whereω representsanindependentsamplefromtheposteriordistribution.MCMC
i
enables sampling from our posterior distribution, with the samples converging to
whentheproductoftheprobabilitydensityandvolumearegreatest[54].
Assumptions previously made in VI methods, such as a factorised posterior
are not required in the MCMC context. MCMC provides convergence to the
true posterior as the number of samples approaches infinity. By avoiding such
restrictions,withenoughtimeandcomputingresourceswecanyieldasolutionthat
isclosertothetruepredictivequantities.ThisisanimportantchallengeforBNNs,
astheposteriordistributionsistypicallyquitecomplex.
TraditionalMCMCmethodsdemonstratearandom-walkbehaviour,inthatnew
proposals in the sequence are generated randomly. Due to the complexity and
highdimensionoftheposteriorinBNNs,thisrandom-walkbehaviourmakesthese
methodsunsuitable for performinginferencein any reasonabletime. To avoid the
random-walkbehaviour,Hybrid/HamiltonianMonteCarlo(HMC)canbeemployed
to incorporate gradient information into the iterative behaviour. While HMC was
initially proposed for statistical physics [57], Neal highlighted the potential for
HMC to address Bayesian inference and specifically researched the applications
toBNNsandthewiderstatisticscommunityasawhole[38].
GiventhatHMCwasinitiallyproposedforphysicaldynamics,itisappropriateto
buildintuitionforappliedstatisticsthroughaphysicalanalogy.Treatourparameters
ofinterestωasapositionvariable.Anauxiliaryvariableisthenintroducedtomodel
themomentumvofourcurrentposition.Thisauxiliaryvariableisnotofstatistical
interest,andisonlyintroducedtoaidindevelopmentofthesystemdynamics.With
a position and momentum variable, we can represent the potential energy U(ω)
and the kinetic energy K(v) of our system. The total energy of a system is then
representedas,
H(ω,v)=U(ω)+K(v). (3.23)
We now consider the case of a lossless system, in that the total energy H(ω,v)
isconstant.15 ThisisdescribedasaHamiltoniansystem,andisrepresentedasthe
followingsystemofdifferentialequations[58],
dw ∂H
i = , (3.24)
dt ∂v
i
dv ∂H
i =− , (3.25)
dt ∂w
i
wheret representstimeandthei denotestheindividualelementsinωandv.
With the dynamics of the system defined, we wish to relate the physical
interpretation to a probabilistic interpretation. This can be achieved through the
15Thevaluesforωandvwillchange,thoughthetotalenergyofthesystemwillremainconstant.

| 62  |     |     | E.GoanandC.Fookes |     |
| --- | --- | --- | ----------------- | --- |
canonicaldistribution,16
|         | (cid:3)     | (cid:4) (cid:3) | (cid:4) (cid:3) | (cid:4)  |
| ------- | ----------- | --------------- | --------------- | -------- |
|         | 1           | 1               |                 |          |
| P(ω,v)= | exp −H(ω,v) | = exp −U(ω)     | exp −K(v)       | , (3.26) |
|         | Z           | Z               |                 |          |
H(ω,v)
where Z is a normalising constant and is our total energy as defined in
Eq. (3.23). From this joint distribution, we see that our position and momentum
variableareindependent.
Our end goal is to find predictive moments and intervals. For a Bayesian this
makesthekeyquantityofinterestthe posteriordistribution.Therefore,wecanset
thepotentialenergywhichwewishtosamplefromto,
|     |           | (cid:10)   | (cid:11) |        |
| --- | --------- | ---------- | -------- | ------ |
|     | U(ω)=−log | p(ω)p(D|ω) | .        | (3.27) |
WithinHMC,thekineticenergycanbefreelyselectedfromawiderangeofsuitable
functions, though is typically chosen such that it’s marginal distribution of v is a
diagonalGaussiancentredattheorigin.
|     |     | K(v)=vTM −1v, |     |     |
| --- | --- | ------------- | --- | --- |
(3.28)
whereMisadiagonalmatrixreferringtothe“mass”ofourvariablesinthisphysical
interpretation. Although this is the most common kinetic energy function used, it
may not be the most suitable. Betancourt [54] surveys the selection the design of
otherGaussiankineticenergieswithanemphasisonthegeometricinterpretations.
Itisalsohighlightedthatselectionofappropriatekineticenergyfunctionsremains
anopenresearchtopic,particularlyinthecaseofnon-Gaussianfunctions.
SinceHamiltoniandynamicsleavesthetotalenergyinvariant,whenimplemented
with infinite precision, the dynamics proposed are reversible. Reversibility is a
sufficient property to satisfy the condition of detailed balance, which is required
to ensure that the target distribution (the posterior we are trying to sample from)
remains invariant. For practical implementations, numerical errors arise due to
discretisationofvariables.Thediscretisationmethodmostcommonlyemployedis
the leapfrogmethod.Theleapfrogmethodspecifiesa step size (cid:9) anda numberof
stepsLtobeusedbeforepossiblyacceptingthenewupdate.Theleapfrogmethod
firstperformsahalfupdateofthemomentumvariablev,followedbyafullupdate
16Asiscommonlydone,weassumethetemperaturevariableincludedinphysicalrepresentations
ofthecanonicaldistributionissettoone.Formoreinformation,see[58,p.11],[59,p.123].

3 BayesianNeuralNetworks:AnIntroductionandSurvey 63
ofthepositionwandthentheremaininghalfupdateofthemomentum[58],
|     | (cid:9) |          | (cid:9)dv i |        |
| --- | ------- | -------- | ----------- | ------ |
|     | v (t +  | )=v (t)+ | (v(t)),     | (3.29) |
|     | i 2     | i        | 2 dt        |        |
dw
|     | +(cid:9))=w | (t)+(cid:9) | i       |        |
| --- | ----------- | ----------- | ------- | ------ |
|     | w i (t      | i           | (w(t)), | (3.30) |
dt
|      |                  | (cid:9) (cid:9)dv | i (cid:9)  |        |
| ---- | ---------------- | ----------------- | ---------- | ------ |
| v (t | +(cid:9))=v (t + | )+                | (v(t + )). | (3.31) |
| i    | i                |                   |            |        |
|      |                  | 2 2               | dt 2       |        |
If the value of (cid:9) is chosen such that this dynamical system remains stable, it can
be shown that this leapfrog method preserves the volume (total energy) of the
Hamiltonian.
Forexpectationstobeapproximatedusing(3.22),werequireeachsampleω
i to
beindependentfromsubsequentsamples.Wecanachievepracticalindependence17
byusingmultipleleapfrogstepsL.Inthisway,afterLleapfrogstepsofsize(cid:9),the
newpositionisproposed.Thisreducescorrelationbetweensamplesandcanallow
for faster exploration of the posterior space. A Metropolis step is then applied to
determinewhetherthisnewproposalisacceptedastheneweststateintheMarkov
Chain[58].
FortheBNNproposedbyNeal[38],ahyper-priorp(γ)isinducedtomodelthe
varianceoverpriorparameterprecisionandlikelihoodprecision.AGaussianprior
isusedforthepriorover-parametersandthelikelihoodissettobeGaussian.There-
γ
fore, the prior over the was Gamma distributed, such that it was conditionally
conjugate.ThisallowsforGibbssamplingtobeusedforperforminginferenceover
hyperparameters.HMC is then used to update the posterior parameters.Sampling
from the joint posterior P(ω,γ|D) then involves alternating between the Gibbs
sampling step for the hyperparameters and Hamiltonian dynamics for the model
parameters. Superior performance of HMC for simple BNN models was then
demonstratedandcomparedwithrandomwalkMCMCandLangevinmethods[38].
3.2.4 Modern BNNs
Considerably less research was conducted into BNNs following early work of
Neal, MacKay and Bishop proposed in the 90s. This relative stagnation was
seen throughout the majority of NN research, and was largely due to the high
computationaldemandfortraining NNs.NNs are parametricmodelsthatare able
to capture any function with arbitrary accuracy, but to capture complex functions
accurately requires large networks with many parameters. Training of such large
networksbecameinfeasibleevenforthetraditionalfrequentistperspective,andthe
17Whereforallpracticalpurposeseachsamplecanbeviewedasindependent.

64 E.GoanandC.Fookes
computational demand significantly increases to investigate the more informative
Bayesiancounterpart.
OnceitwasshownthatgeneralpurposeGPUscouldaccelerateandallowtraining
of large models, interest and research into NNs saw a resurgence. GPUs enabled
large scale parallelism of the linear algebra performed during back propagation.
This accelerated computation has allowed for training of deeper networks, where
successiveconcatenationofhiddenlayersisused.WiththeproficiencyofGPUsfor
optimisingcomplexnetworksandthegreatempiricalsuccessseenbysuchmodels,
interestintoBNNsresumed.
ModernresearchintoBNNshaslargelyfocusedontheVIapproach,giventhat
theseproblemscanbeoptimisedusingasimilarbackpropagationapproachusedfor
pointestimatenetworks.Giventhatthenetworksofferingthemostpromisingresults
use multiplelayers,the originalVI approachesshownin [52,53], whichfocuson
analytical approximations for regression networks utilising a single hidden layer
became unsuitable. Modern NNs now exhibit considerably different architectures
withvaryingdimensions,hiddenlayers,activationsandapplications.Moregeneral
approachesforviewingnetworksinaprobabilisticsensewasrequired.
Giventhelargescaleofmodernnetworks,largedatasetsaretypicallyrequired
for robust inference.18 For these large data sets, evaluation of the complete log-
likelihood becomes infeasible for training purposes. To combat this, a Stochastic
GradientDescent(SGD)approachisused,wheremini-batchesofthedataareused
toapproximatethelikelihoodterm,suchthatourvariationalobjectivebecomes,
(cid:2)N (cid:3) (cid:4) (cid:10) (cid:11)
N
L(ω,θ)=− E
q
[log p(D
i
|ω) ]+KL qθ(ω)||p(ω) , (3.32)
M
i=1
whereD ⊂D,andeachsubsetisofsizeM.Thisprovidesanefficientwaytoutilise
i
largedatasetsduringtraining.AfterpassingasinglesubsetD ,backpropagationis
i
appliedtoupdatethemodelparameters.Thissub-samplingofthelikelihoodinduces
noiseintoourinferenceprocess,hencethenameSGD.Thisnoisethatisinducedis
expectedtoaverageoutoverevaluationofeachindividualsubset[60].SGDisthe
mostcommonmethodfortrainingNNsandBNNsutilisingaVIapproach.
A keypaperin the resurgenceofBNN researchwas publishedby Graves[61].
This work proposes a MFVB treatment using a factorised Gaussian approximate
posterior. The key contributionof this work is the computationof the derivatives.
TheVIobjective(ELBO)canbeviewedasasumoftwoexpectations,
(cid:3) (cid:4)
F[qθ ]=E
q
[log p(D|ω) ]−E
q
[logqθ(ω)−logp(ω)] (3.33)
18Neal [38] argues that this not true for Bayesian modelling; claims that if suitable prior
informationisavailable,complexityofamodelshouldonlybelimitedbycomputationalresources.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 65
It is these two expectations that we need to optimise w.r.t model parameters,
meaningthatwe requirethe gradientofexpectations.Thisworkshowshowusing
the gradient properties of a Gaussian proposed in [62] can be used to perform
parameterupdates,
| ∇ E | [f(ω)]=E | [∇ ωf(ω)], |     | (3.34) |
| --- | -------- | ---------- | --- | ------ |
| μ   | p(ω)     | p(ω)       |     |        |
1
| ∇ E | p(ω) [f(ω)]= | E p(ω) [∇ ω ∇ ωf(ω)]. |     | (3.35) |
| --- | ------------ | --------------------- | --- | ------ |
Σ
2
MCintegrationcouldbeappliedto Eqs.(3.34)and(3.35)to approximatethegra-
dientofthemeanandvarianceparameters.Thisframeworkallowsforoptimisation
oftheELBOtogeneralisetoanylog-lossparametricmodel.
Whilst addressing the problem of applying VI to complex BNNs with more
hiddenlayers,practicalimplementationshaveshowninadequateperformancewhich
is attributed to large variance in the MC approximations of the gradient compu-
tations [63]. Developing gradient estimates with reduced variance has become an
integralresearch topic in VI [64]. Two of the mostcommonmethodsfor deriving
gradientapproximationsrelyontheuseofscorefunctionsandpath-wisederivative
estimators.
Scorefunctionestimatorsrelyontheuseofthelog-derivativeproperty,suchthat,
| ∂   |               | ∂          |     |        |
| --- | ------------- | ---------- | --- | ------ |
|     | p(x|θ)=p(x|θ) | logp(x|θ). |     | (3.36) |
| ∂θ  |               | ∂θ         |     |        |
Using this property, we can form Monte Carlo estimates of the derivatives of an
expectation,whichisoftenrequiredinVI,
(cid:7)
| ∇ E [f(ω)]= | f(ω)∇ |             |     |     |
| ----------- | ----- | ----------- | --- | --- |
| θ q         |       | θ q θ (ω)∂ω |     |     |
(cid:7)
(cid:10) (cid:11)
|     | =        | (ω)∇     |            |     |
| --- | -------- | -------- | ---------- | --- |
|     | f(ω)q    | θ θ log  | q θ (ω) ∂ω |     |
|     | (cid:2)L | (cid:10) | (cid:11)   |     |
1
|     | ≈   | f(ω )∇ log q | (ω ) . | (3.37) |
| --- | --- | ------------ | ------ | ------ |
|     |     | i θ          | θ i    |        |
L
i=1
A common problem with score function gradient estimators is that they exhibit
considerablevariance[64].Oneofthemostcommonmethodstoreducethevariance
inMonteCarloestimatesistheintroductionofcontrolvariates[65].
ThesecondtypeofgradientestimatorcommonlyusedintheVIliteratureisthe
pathwise derivative estimator. This work builds on the “reparameterisation trick”
[66–68],wherearandomvariableisrepresentedasadeterministicanddifferentiable

66 E.GoanandC.Fookes
expression.Forexample,foraGaussianwithθ ={μ,σ},
ω∼N(μ,σ2)
|     | ω=g(θ,(cid:7))=μ+σ |     | (cid:8)(cid:7) |     |     |
| --- | ------------------ | --- | -------------- | --- | --- |
(3.38)
where (cid:7) ∼ N(0,I) and (cid:8) represents the Hadamard product. Using this method
allows for efficient sampling for Monte Carlo estimates of expectations. This is
shownin[67],thatwithω=g(θ,(cid:7)),weknowthatq(ω|θ)dω=p((cid:7))d(cid:7).Therefore,
wecanshowthat,
| (cid:7) |              | (cid:7) |                        |     |     |
| ------- | ------------ | ------- | ---------------------- | --- | --- |
|         | qθ(ω)f(ω)dω= |         | p((cid:7))f(ω)d(cid:7) |     |     |
(cid:7)
= p((cid:7))f(g(θ,(cid:7)))d(cid:7)
| (cid:2)M |               |       | (cid:2)M |                |        |
| -------- | ------------- | ----- | -------- | -------------- | ------ |
| ≈ 1      | f(g(θ,(cid:7) | ))= 1 | f(μ+σ    | (cid:8)(cid:7) |        |
|          |               |       |          | )              | (3.39) |
| M        |               | i M   |          | i              |        |
|          | i=1           |       | i=1      |                |        |
θ,
Since Eq. (3.39)is differentiablew.r.t gradientdescent methodscan be used to
optimisethisexpectationapproximation.ThisisanimportantpropertyinVI,since
theVIobjectivecontainsexpectationsofthelog-likelihoodthatareoftenintractable.
The reparameterisation trick serves as the basis for pathwise-gradient estimators.
Pathwise estimators are favourable for their reduced variance over score function
estimators[64,67].
A key benefit of having a Bayesian treatment of NNs is the ability to extract
uncertainty in our models and their predictions. This has been a recent research
topic of high interest in the context of NNs. Promising developments regarding
uncertainty estimation in NNs has been found by relating existing regularisation
techniquessuchasDropout[69]toapproximateinference.DropoutisaStochastic
RegularisationTechnique(SRT)thatwasproposedtoaddressoverfittingcommonly
seen in point-estimate networks. During training, Dropoutintroducesan indepen-
dent random variable that is Bernoulli distributed, and multiplies each individual
weightelement-wisebyasamplefromthisdistribution.Forexample,asimpleMLP
implementingDropoutisoftheform,
ρ ∼Bernoulli(p),
u
|     |      | (cid:10)(cid:2)N1 | (cid:11) |     |        |
| --- | ---- | ----------------- | -------- | --- | ------ |
|     | φ =θ | (x                | ρ )w .   |     | (3.40) |
|     | j    | i                 | u ij     |     |        |
i=1
Looking at Eq. (3.40), it can be seen that the application of Dropout introduces
stochasticity into the network parameters in a similar manner as to that of the
reparameterisationtrickshowninEq.(3.38).Akeydifferenceisthatinthecaseof

3 BayesianNeuralNetworks:AnIntroductionandSurvey 67
Dropout,stochasticityisintroducedintotheinputspace,asapposetotheparameter
space required for Bayesian inference. Yarin Gal [70] identified this similarity,
anddemonstratedhownoiseintroducedthroughtheapplicationofDropoutcanbe
transferredtothenetworksweightsefficientlyas,
|     | W1 =diag(ρ)W1 |     |     | (3.41) |
| --- | ------------- | --- | --- | ------ |
ρ
(cid:3) (cid:4)
|     | (cid:2) =a XTW1 | .   |     | (3.42) |
| --- | --------------- | --- | --- | ------ |
|     | ρ               | ρ   |     |        |
Where ρ is a vector sampled from the Bernoulli distribution, and the diag(·)
operator creates a square diagonal matrix from a vector. In doing this it can be
seenthatasingledropoutvariableissharedamongsteachrowoftheweightmatrix,
allowingsomecorrelationwithinrowstobemaintained.Byviewingthestochastic
component in terms of network weights, the formulation becomes suitable for
approximate inference using the VI framework. In this work, the approximate
posterior is of the form of a Bernoulli distribution multiplied by the network
weights.
Thereparameterisationtrickisthenappliedtoallowforpartialderivativesw.r.t.
network parameters to be found. The ELBO is then formed and backpropagation
isperformedtomaximisethelowerbound.MCintegrationisusedtoapproximate
the analytically intractable expected log-likelihood. The KL divergence between
the approximate posterior and the prior distribution in the ELBO is then found
by approximatingthe scaled Bernoulli approximate posterior as a mixture of two
Gaussianswithverysmallvariance.
In parallel to this work, Kingma et al. [71] identified this same similarity
betweenDropoutandit’spotentialforusewithinaVIframework.Asapposetothe
typicalBernoullidistributedr.v.introducedinDropout,Kingmaetal.[71]focuses
attentiontothecasewhentheintroducedr.v.isGaussian[72].Itisalsoshownhow
with selection of an appropriate prior that is independent of parameters, current
applicationsofNNsusingdropoutcanbeviewedasapproximateinference.
Kingmaetal. alsoaimstoreducethevarianceinthestochastic gradientsusing
a refined, local reparameterisation. This is done by instead of sampling from
the weight distribution before applying the affine transformation, the sampling is
performed afterwards. For example, consider a MFVB case where each weight
is assumed to be an independent Gaussian w ∼ N(μ ,σ2). After the affine
| (cid:14) |     | ij  | ij ij |     |
| -------- | --- | --- | ----- | --- |
transformation φ = N1 (x ρ )w , the posterior distribution of φ conditional
| j i=1 | i i ij |     | j   |     |
| ----- | ------ | --- | --- | --- |
ontheinputswillalsobeafactorisedGaussian,
|     | q(φ |x)=N(γ | ,δ2), |     | (3.43) |
| --- | ----------- | ----- | --- | ------ |
|     | j           | j j   |     |        |
(cid:2)N
|     | γ = | x μ , |     | (3.44) |
| --- | --- | ----- | --- | ------ |
|     | j   | i i,j |     |        |
i=1
(cid:2)N
|     | δ2 = | x2σ2 . |     | (3.45) |
| --- | ---- | ------ | --- | ------ |
|     | j    | i i,j  |     |        |
i=1

68 E.GoanandC.Fookes
Itisadvantageoustosamplefromthisdistributionforφasapposetothedistribution
oftheweightsw themselves,asthisresultsinagradientestimatorwhosevariance
scaleslinearlywiththenumberofmini-batchesusedduringtraining.19
These few bodies of work are important in addressing the serious lack of
rigour seen in ML research. For example, the initial Dropout paper [69] lacks
anysignificanttheoreticalfoundation.Instead,themethodcitesatheoryforsexual
reproduction[73]asmotivationforthemethod,andreliesheavilyontheempirical
results given. These empirical results have been further demonstrated throughout
many high impact20 research items which utilise this technique merely as a
regularisation method. The work in [70] and [71] show that there is theoretical
justificationforsuchanapproach.Inattemptstoreducetheeffectofoverfittingina
network,thefrequentistmethodologyreliedontheapplicationofaweaklyjustified
technique that shows empirical success, while Bayesian analysis provides a rich
bodyoftheorythatnaturallyleadstoameaningfulunderstandingofthispowerful
approximation.
Whilst addressing the problem of applying VI to complex BNNs with more
hiddenlayers,practicalimplementationshaveshowninadequateperformancewhich
is attributed to large variancein the MC approximationsof the gradientcomputa-
tions.Hernandezetal.[63]acknowledgethislimitationandproposeanewmethod
for practicalinferenceof BNNs titled Probabilistic Back Propagation(PBP). PBP
deviates from the typical VI approach, and instead employs an Assumed Density
Filtering(ADF)method[74].Inthisformat,theposteriorisupdatedinaniterative
fashionthroughapplicationofBayesrule,
|     | p(ω   | |D ) p ( D | |ω ) |        |
| --- | ----- | ---------- | ---- | ------ |
| p(ω | |D )= | t t t +1   | t    |        |
| t+1 | t+1   |            | .    | (3.46) |
p ( D )
t + 1
Asopposedtotraditionalnetworktrainingwherethepredictederroristheobjective
function, PBP uses a forward pass to compute the log-marginal probability of a
target and updates the posterior distribution of network parameters. The moment
matchingmethoddefinedin[75]updatestheposteriorusingavariantofbackprop-
agation,whilstmaintainingequivalentmeanandvariancebetweentheapproximate
andvariationaldistribution,
∂logp(D
t+1 )
| μ =μ  | +σ  |     |     | (3.47) |
| ----- | --- | --- | --- | ------ |
| t+1 t | t   |     |     |        |
∂μ
|       | (cid:8)(cid:3) D | (cid:4)   | D (cid:9) |        |
| ----- | ---------------- | --------- | --------- | ------ |
|       | ∂p(              | t+1 ) ∂p( | t+1 )     |        |
| σ =σ  | +σ 2             | 2−2       | .         | (3.48) |
| t+1 t | t                |           |           |        |
|       | ∂ μ              | t         | ∂ σ       |        |
19Thismethodalsohascomputationaladvantages,asthedimensionofφistypicallymuchlower
thanthatofω.
20Atthetimeofwriting,[69]hasovertenthousandcitations.

| 3 BayesianNeuralNetworks:AnIntroductionandSurvey |     |     |     | 69  |
| ------------------------------------------------ | --- | --- | --- | --- |
Experimental results on multiple small data-sets illustrate reasonable perfor-
mance in terms of predicted accuracy and uncertainty estimation when compared
with HMC methods for simple regression problems[63]. A key limitation of this
methodis the computationalbottleneck introducedby the onlinetraining method.
Thisapproachmaybesuitableforsomeapplications,orforupdatingexistingBNNs
with additional data as it becomes available, though for performing inference on
largedatasetsthemethodiscomputationallyprohibitive.
A promising method for approximate inference in BNNs was proposed by
Blundelletal.,titled“BayesbyBackprop”[76].Themethodutilisesthereparam-
eterisationtricktoshowhowunbiasedestimatesofthederivativeofanexpectation
can be found. For a random variable ω ∼ qθ(ω) that can be reparameterised
ω = g((cid:7),θ),
as deterministic and differentiable function the derivative of the
expectationofanarbitraryfunctionf(ω,θ)canbeexpressedas,
(cid:7)
| ∂   |             | ∂             |     |        |
| --- | ----------- | ------------- | --- | ------ |
|     | E [f(ω,θ)]= | qθ(ω)f(ω,θ)dω |     |        |
|     | q           |               |     | (3.49) |
| ∂θ  |             | ∂θ            |     |        |
(cid:7)
∂
= p((cid:7))f(ω,θ)d(cid:7)
(3.50)
∂θ
|     |     | (cid:8)           | (cid:9) |     |
| --- | --- | ----------------- | ------- | --- |
|     |     | ∂f(ω,θ)∂ω ∂f(ω,θ) |         |     |
=E +
|     |     | q((cid:9)) | .   | (3.51) |
| --- | --- | ---------- | --- | ------ |
|     |     | ∂ω ∂θ      | ∂θ  |        |
IntheBayesbyBackpropalgorithm,thefunctionf(ω,θ)issetas,
qθ(ω)
f(ω,θ)=log −logp(X|ω).
(3.52)
p(ω)
f(ω,θ)
This can be seen as the argument for the expectation performed in
Eq.(3.17),whichispartofthelowerbound.
CombiningEqs.(3.51)and(3.52),
|     |     | (cid:15) | (cid:16) |     |
| --- | --- | -------- | -------- | --- |
qθ(ω)
| L(ω,θ)=E | [f(ω,θ)]=e | log −logp(D|ω) | =−F[qθ ] | (3.53) |
| -------- | ---------- | -------------- | -------- | ------ |
|          | q          | q p(ω)         |          |        |
whichisshowntobethenegativeoftheELBO,meaningthatBayesbyBackprop
aims to minimise the KL divergencebetween the approximate and true posterior.
MonteCarlointegrationisused21toapproximatethecostinEq.(3.53),
(cid:2)N ω
q θ ( i )
|     | F[qθ | ]≈ log −logp(X|ω | )   | (3.54) |
| --- | ---- | ---------------- | --- | ------ |
|     |      | p ( ω )          | i   |        |
i
i=1
21Sometermsmaybetractableinthisintegrand,dependingontheformofthepriorandposterior
approximation.MCintegrationallowsforarbitrarydistributionstobeapproximated.

| 70  |     |            |     | E.GoanandC.Fookes |     |
| --- | --- | ---------- | --- | ----------------- | --- |
| ω   | ith | fromqθ(ω). |     |                   |     |
where i is the sample With the approximationin Eq. (3.54), the
unbiasedgradientscanbefoundusingtheresultshowninEq.(3.51).
For the Bayes by Backprop algorithm, a fully factorised Gaussian posterior is
|              | θ =  | {μ,ρ}, | σ = softplus(ρ) |            |            |
| ------------ | ---- | ------ | --------------- | ---------- | ---------- |
| assumed such | that | where  |                 | is used to | ensure the |
standard deviation parameter is positive. With this, the distribution of weights
ω∼N(μ,softplus(ρ)2)inthenetworkarereparameterisedas,
|     | ω=g(θ,(cid:7))=μ+softplus(ρ)(cid:8)(cid:7). |     |     |     | (3.55) |
| --- | ------------------------------------------- | --- | --- | --- | ------ |
In this BNN, the trainable parameters are μ and ρ. Since a fully factorised
distribution is used, following from Eq. (3.20), the logarithm of the approximate
posteriorcanberepresentedas,
|     |           | (cid:2) | (cid:10)  | (cid:11) |        |
| --- | --------- | ------- | --------- | -------- | ------ |
|     | logqθ(ω)= | log     | N(w ;μ ,σ | 2 ) .    | (3.56) |
|     |           |         | ljk ljk   | l jk     |        |
l,j,k
ThecompleteBayesbyBackpropalgorithmisdescribedinAlgorithm1.
Algorithm1BayesbyBackprop(BbB)algorithm[76]
1: procedureBBB(θ,X,α)
2: repeat
| 3: F[qθ              | ]←0                   |     |                                      | (cid:10)Initialisecost |     |
| -------------------- | --------------------- | --- | ------------------------------------ | ---------------------- | --- |
| 4: foriin[1,...,N]do |                       |     | (cid:10)NumberofsamplesforMCestimate |                        |     |
| 5:                   | Sample(cid:7) ∼N(0,1) |     |                                      |                        |     |
i
| 6:  | ω←μ+softplus(ρ)·(cid:7) |     |     |     |     |
| --- | ----------------------- | --- | --- | --- | --- |
i
| 7:  | L←logq(ω|θ)−logp(ω)−logp(X|ω) |     |                                        |     |     |
| --- | ----------------------------- | --- | -------------------------------------- | --- | --- |
| 8:  | F[qθ ]+=sum(L)/N              |     | (cid:10)Sumacrossalllogofweightsinsetω |     |     |
9: endfor
| 10: θ | ←θ−α∇ F[qθ ] |     |     | (cid:10)Updateparameters |     |
| ----- | ------------ | --- | --- | ------------------------ | --- |
θ
11: untilconvergence
12: endprocedure
| 3.2.5 GaussianProcess |     | Properties | ofBNNs |     |     |
| --------------------- | --- | ---------- | ------ | --- | --- |
Neal[38]alsoprovidedderivationandexperimentationresultstoillustratethatfora
networkwithasinglehiddenlayer,aGaussianProcess(GP)prioroverthenetwork
outputariseswhenthenumberofhiddenunitsapproachesinfinity,andaGaussian
priorisplacedoverparameters.22Figure3.6illustratesthisresult.
22Foraregressionmodelwithnonon-linearactivationfunctionplacedontheoutputunits.

| 3 BayesianNeuralNetworks:AnIntroductionandSurvey |     |     |     |     | 71  |
| ------------------------------------------------ | --- | --- | --- | --- | --- |
| 6                                                |     |     | 5   |     |     |
4
4
3
| 2   |     |     | 2   |     |     |
| --- | --- | --- | --- | --- | --- |
1
0
0
| -2  |     |     | -1  |     |     |
| --- | --- | --- | --- | --- | --- |
-2
-4
-3
| -6    |         |       | -4         |         |         |
| ----- | ------- | ----- | ---------- | ------- | ------- |
| -6 -4 | -2 0    | 2 4   | 6 -4 -3 -2 | -1 0 1  | 2 3 4 5 |
|       | (a)     |       |            | (b)     |         |
| 4     |         |       | 4          |         |         |
| 3     |         |       | 3          |         |         |
| 2     |         |       | 2          |         |         |
| 1     |         |       | 1          |         |         |
| 0     |         |       | 0          |         |         |
| -1    |         |       | -1         |         |         |
| -2    |         |       | -2         |         |         |
| -3    |         |       | -3         |         |         |
| -4 -3 | -2 -1 0 | 1 2 3 | 4 -4 -3    | -2 -1 0 | 1 2 3 4 |
|       | (c)     |       |            | (d)     |         |
Fig.3.6 IllustrationofGPpriorinducedonoutputwhenplacingaGaussianprioroverparameters
asthenetworksizeincreases.Experimentationreplicatedfrom[38,p.33].Eachdotcorresponds
totheoutputofanetworkwithparameterssampledfromtheprior,withthex-axisasf(0.2)and
they-axisasf(−0.4).Foreachnetwork,thenumberofhiddenunitsare(a)1,(b)3,(c)10,(d)
100
ThisimportantlinkbetweenNNsandGPscanbeseenfromEqs.(3.1)and(3.2).
Fromtheseexpressions,itcanbeseenthataNNwithasinglehiddenlayerisasum
ofN parametricbasisfunctionsappliedtotheinputdata.Iftheparametersforeach
basisfunctioninEq.(3.1)arer.v.’s,Eq.(3.2)becomesthesumofr.v.’s.Underthe
centrallimittheorem,asthenumberofhiddenlayersN →∞,theoutputbecomes
Gaussian. Since the outputis then describedas an infinite sum of basis functions,
the output can be seen to become a GP. Following from a full derivation of this
resultand the illustrations show in Fig. 3.6,Neal [38] showshow an approximate
Gaussiannatureisachievedforfinitecomputingresourcesandhowthemagnitude
of this sum can be maintained. Williams then demonstrated how the form of the
covariance function could be analysed for differentactivation functions [77]. The
relationbetweenGPsandinfinitelywidenetworkswithasinglehiddenlayerwork
hasrecentlybeenextendedtothecaseofdeepnetworks[78].

72 E.GoanandC.Fookes
Identification of this link has motivated many research works in BNNs. GPs
provide many of the properties we wish to obtain, such as reliable uncertainty
estimates, interpretabilityand robustness. GPs deliver these benefits at the cost of
predictive performance and exponentially large computational resources required
as the size of data sets increase. This link between GPs and BNNs has motivated
themergingofthetwomodellingschemes;maintainingthepredictiveperformance
and flexibility seen in NNs while incorporating the robustness and probabilistic
propertiesenabledbyGPs. Thishasled to thedevelopmentof the DeepGaussian
Process.
Deep GPs are a cascade of individual GPs, where much like a NN, the output
of the previous GP serves as the input to a new GP [79, 80]. This stacking of
GPs allows for learning of non-Gaussian densities from a combination of GPs.23
A key challenge with GPs is fitting to large data sets, as the dimensions of the
Gram matrix for a single GP is quadratic with the number of data points. This
issueisamplifiedwithaDeepGP,aseachindividualGPinthecascadeinducesan
independentGrammatrix.Furthermore,the marginallikelihoodforDeep GPsare
analytically intractable due to non-linearities in the functions produced. Building
on the work in [82], Damianou and Lawrence [79] use a VI approach to create
an approximation that is tractable and reduces computational complexity to that
typicallyseeninsparseGPs[83].
DeepGPshaveshownhowtheGPscanbenefitfrommethodologyseeninNNs.
Gal and Ghahramani [84–86] built of this work to show how a Deep GP can
be approximated with a BNN.24 This is an expected result; given that Neal [38]
identified an infinitely wide network with a single hidden layer converges to a
Gaussianprocess,byconcatenatingmultipleinfinitelywidelayersweconvergetoa
deepGaussianprocess.
Alongsidethis analysisofdeepGaussian processes, [84–86] buildon the work
in [77] to analyse the relationship between the modern non-linear activation used
within BNNs and the covariance function for a GP. This is promising work that
couldallowformoreprincipledselectionofactivationfunctionsinNNs,similarto
thatofGPs.Whichactivationfunctionswillyieldastationaryprocess?Whatisthe
expectedlengthscaleforourprocess?Thesequestionsmaybeabletobeaddressed
usingtherichtheoryexistingforGPs.
TheGPpropertiesarenotrestrictedtoMLPBNNs.Recentresearchhasidenti-
fiedcertainrelationshipsandconditionsthatinduceGPpropertiesinconvolutional
BNNs [87,88]. This resultis expectedsince CNNs can be implementedas MLPs
with structureenforcedin the weights.What thisworkidentifiesis how the GPis
constructedwhen this structure is enforced.Van der Wilk et al. [89] proposedthe
ConvolutionalGaussianProcess,whichimplementsapatchbasedoperationsimilar
tothatseeninCNNstodefinetheGPprioroverfunctions.Practicalimplementation
23AcompleteintroductiontoDeepGPs,alongwithcodeandlectureshasbeenoffered byNeil
Lawrence[81].
24ApproximationbecomesaDeepGPasthenumberofhiddenunitsineachlayerapproaches∞.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 73
ofthismethodrequirestheuseofapproximationmethods,duetotheprohibitivecost
ofevaluatinglargedatasets,andevenevaluationateachpatch.Inducingpointsare
formedwithaVIframeworktoreducethenumberofdatapointstoevaluateandthe
numberofpatchesevaluated.
3.2.6 Limitationsin CurrentBNNs
WhilstgreatefforthasbeenputintodevelopingBayesianmethodsforperforming
inference in NNs, there are significant limitations to these methods and many
gaps remaining in the literature. A key limitation is the heavy reliance on VI
methods.WithintheVIframework,themostcommonapproachistheMeanField
approach.MFVBprovidesaconvenientwaytorepresentanapproximateposterior
distributionbyenforcingstrongassumptionsofindependencebetweenparameters.
This assumption allows for factorised distributions to be used to approximate the
posterior.Thisassumptionofindependencesignificantlyreducesthecomputational
complexityofapproximateinferenceatthecostofprobabilisticaccuracy.
AcommonfindingwithVIapproachesisthatresultingmodelsareoverconfident,
in that predictive means can be accurate while variance is considerably under
estimated[50,90–93]. Thisphenomenonisdescribedin Section 10.1.2of [2] and
Section 21.2.2 of [35], both of which are accompanied by examples and intuitive
figures to illustrate this property. This property of under-estimated variance is
presentwithinmuchofthecurrentresearchinBNNs[70].Recentworkhasaimed
toaddresstheseissuesthroughtheuseofnoisecontrastivepriors[94]andthrough
useofcalibrationdatasets[95].Theauthorsin[96]employtheuseoftheconcrete
distribution[97]toapproximatetheBernoulliparameterintheMCDropoutmethod
[85],allowingforittobeoptimised,resultinginposteriorvariancesthatarebetter
calibrated. Despite these efforts, the task of formulating reliable and calibrated
uncertaintyestimateswithinaVIframeworkforBNNsremainsunsolved.
It is reasonable to consider that perhaps the limitations of the current VI
approaches are influenced by the choice of approximate distribution used, partic-
ularlytheusualMFVBapproachofindependentGaussians.Ifmorecomprehensive
approximate distributions are used, will our predictions be more consistent with
the data we have and haven’t seen? Mixture based approximations have been
proposedforthe generalVIapproach[48,98],thoughintroductionofN mixtures
increasesthenumberofvariationalparametersbyN. Matrix-Normalapproximate
posteriors have been introduced to the case of BNNs [99], which reduces the
number of variational parameters in the model when compared with a full rank
Gaussian, though this work still factorises over individual weights, meaning no
covariancestructureismodelled.25MCDropoutisabletomaintainsomecorrelation
25Though this work highlights that even with a fully factorised distribution over weights, the
outputsofeachlayerwillbecorrelated.

74 E.GoanandC.Fookes
informationwithintherowsofweightmatrix,atthecompromiseofalowentropy
approximateposterior.
ArecentapproachforVIhasbeenproposedtocapturemorecomplexposterior
distributionsthroughtheuseofnormalisingflows[100,101].Withinanormalising
flow, the initial distribution “flows” through a sequence of invertible functions to
produceamorecomplexdistribution.ThiscanbeappliedwithintheVIframework
using amortized inference [102]. Amortized inference introduces an inference
networkwhichmapsinputdatato the variationalparametersof generativemodel.
These parameters are then used to sample from the posterior of the generative
process.TheuseofnormalisingflowshasbeenextendedtothecaseofBNNs[103].
Issues arise with this approach relating to the computational complexity, along
withlimitationsofamortizedinference.Normalisingflowsrequiresthecalculation
of the determinant of the Jacobian for applying the change of variables used
for each invertible function, which can be computationally expensive for certain
models. Computational complexity can be reduced by restricting the normalising
flow to contain invertibleoperationsthat are numericallystable [102,104]. These
restrictionshavebeenshowntoseverelylimittheflexibilityoftheinferenceprocess,
andthecomplexityoftheresultingposteriorapproximation[105].
Asstatedpreviously,intheVIframework,anapproximatedistributionisselected
and the ELBO is then maximised. This ELBO arises from the applying the KL
divergencebetweenthetrueandapproximateposterior,butthisbegsthequestion,
whyusetheKL?TheKLdivergenceisawellknownmeasuretoassessthesimilarity
of between two distributions, and satisfies all the key properties of a divergence
(i.e. is positive and only zero when the two distributionsare equal). A divergence
allowsustoknowwhetherourapproximationisapproachingthetruedistribution,
butnothowclosewearetoit.Whynotuseofawelldefineddistanceasapposeto
adivergence?
The KL divergence is used as it allows us to separate the intractable quantity
(the marginallikelihood)out of our objective function (the ELBO) which we can
optimise. Our goal with our Bayesian inference is to identify the parameters that
bestfitourmodelunderpriorknowledgeandthedistributionoftheobserveddata.
TheVIframeworkposesinferenceasanoptimisationproblem,whereweoptimise
our parametersto minimise the KL divergencebetween our approximateand true
distribution(whichmaximisesourELBO).Sinceweareoptimisingourparameters,
by separating the marginallikelihood from our objective function, we are able to
compute derivatives with respect to the tractable quantities. Since the marginal
likelihood is independent of the parameters, this component vanishes when the
derivativeistaken.ThisisthekeyreasonwhytheKLdivergenceisused,asitallows
ustoseparatetheintractablequantityoutofourobjectivefunction,whichwillthen
beevaluatedaszerowhenusinggradientinformationtoperformoptimisation.

| 3   | BayesianNeuralNetworks:AnIntroductionandSurvey |     |     |     |     |     |     | 75  |
| --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
TheKLdivergencehasbeenshowntobepartofagenericfamilyofdivergences
knownasα-divergences[106,107].Theα-divergenceisrepresentedas,
(cid:7)
|     |     |     |     | (cid:10) |     |     | (cid:11) |     |
| --- | --- | --- | --- | -------- | --- | --- | -------- | --- |
1
|     | D [p(ω)||q(ω)]= |     |        | 1−  | p(ω)αq(ω)1−αdω |     | .   | (3.57) |
| --- | --------------- | --- | ------ | --- | -------------- | --- | --- | ------ |
|     | α               |     | α(1−α) |     |                |     |     |        |
The forward KL divergence used in VI is found from Eq. (3.57) in the limit that
| →   | −1,    |           | divergenceKL(p||q) |     |          |     |             | →    |
| --- | ------ | --------- | ------------------ | --- | -------- | --- | ----------- | ---- |
| α   | andthe | reverseKL |                    |     | occursin |     | the limitof | α 1, |
which is used during expectation propagation. While the use of the forward KL
divergenceusedinVItypicallyresultsinanunder-estimatedvariance,theuseofthe
reverseKL will often over-estimatevariance[2]. Similarly,the Hellinger distance
| arisesfrom(3.57)whenα |     |     | =0, |                  |     |          |     |     |
| --------------------- | --- | --- | --- | ---------------- | --- | -------- | --- | --- |
|                       |     |     |     | (cid:7) (cid:10) |     | (cid:11) |     |     |
2
|     |     | (p(ω)||q(ω))2 |     | =    | 1 −q(ω) | 1   |     |        |
| --- | --- | ------------- | --- | ---- | ------- | --- | --- | ------ |
|     |     | D H           |     | p(ω) | 2       | 2   | dω. | (3.58) |
Thisisavaliddistance,inthatitsatisfiesthetriangleinequalityandissymmetric.
Minimisation of the Hellinger distance has shown to provide reasonable compro-
mise in variance estimate when compared with the two KL divergences [107].
Though these measures may provide desirable qualities, they are not suitable for
direct use within VI, as the intractable marginal likelihood cannot be separated
from the other terms of interest.26 While these measures cannot be immediately
used, it illustrates how a change in the objective measure can result in different
approximations.Itispossiblethatmoreaccurateposteriorexpectationscanbefound
byutilisingadifferentmeasurefortheobjectivefunction.
The vast majority of modern works have revolved around the notion of VI.
This is largely due to its amenability to SGD. Sophisticated tools now exist to
simplify and accelerate the implementationof automatic differentiationand back-
propagation [108–114]. Another benefit of VI is it’s acceptance of sub-sampling
inthelikelihood.Sub-samplingreducesthecomputationalexpenseforperforming
inference required to train over large data sets currently available. It is this key
reasonthatmoretraditionalMCMCbasedmethodshavereceivedsignificantlyless
attentionintheBNNcommunity.
MCMC serves as the gold standard for performing Bayesian inference due to
it’srichtheoreticaldevelopment,asymptoticguaranteesandpracticalconvergence
diagnostics. Traditional MCMC based methods require sampling from the full
joint likelihood to perform updates, requiring all training data to be seen before
any new proposal can be made. Sub-sampling MCMC, or Stochastic Gradient
MCMC (SG-MCMC) approaches have been proposed in [60, 115, 116], which
have since been applied to BNNs [117]. It has since been shown that the naive
sub-samplingwithinMCMCwillbiasthetrajectoryofthestochasticupdatesaway
26ThismaybeeasytoseefortheHellingerdistance,butlesssoforthereverse KLdivergence.
Enthusiasticreadersareencouragedtonottakemywordforit,andtoputpenandpapertoprove
thisforthemselves!

76 E.GoanandC.Fookes
fromtheposterior[118].Thisbiasremovesthetheoreticaladvantagesgainedfroma
traditionalMCMCapproach,makingthemlessdesirablethanaVIapproachwhich
isoftenlesscomputationallyexpensive.Forsamplingmethodstobecomefeasible,
sub-samplingmethodsneedtobedevelopedthatassureconvergencetotheposterior
distribution.
3.3 Comparison ofModern BNNs
From the literature survey presented within, two prominent methods for approx-
imate inference in BNNs was Bayes by Backprop and MC Dropout [85]. These
methods have found to be the most promising and highest impact methods for
approximateinferenceinBNNs.ThesearebothVImethodsthatareflexibleenough
to permit the use of SGD, making deployment to large and practical data sets
feasible. Given their prominence, it is worthwhile to compare the methods to see
howwelltheyperform.
To compare these methods, a series of simple homoskedastic regression tasks
were conducted. For these regression models, the likelihood is represented as
Gaussian.Withthiswecanwritethattheun-normalisedposterioris,
p(ω|D)∝p(ω)N(f ω (D),σ2I), (3.59)
where f ω (D) is the function represented by the BNNs. A mixture of Gaussians
was used to model a spike-slab prior for both models. The approximateposterior
qθ(ω) was then foundforeach modelusing the respectivemethodsproposed.For
Bayes by Backprop, the approximateposterior is a fully factorised Gaussian, and
forMCDropoutisascaledBernoullidistribution.With theapproximateposterior
foreachmodel,predictivequantitiescanbefoundusingMCIntegration.Thefirst
twomomentscanbeapproximatedas[70],
(cid:2)N
1
E [y ∗]≈ f ω i(x ∗ ) (3.60)
q
N
i=1
(cid:2)N
1
E [y ∗Ty ∗]≈σ2I+ f ω i(x ∗ )Tf ω i(x ∗ ) (3.61)
q
N
i=1
∗ ∗
wherethestarsuperscriptdenotesthenewinputandoutputsamplex ,y fromthe
testset.
The data sets used to evaluate these models were simple toy data sets from
high impact papers, where similar experimentation was provided as empirical
evidence [76, 119]. Both BNN methods were then compared with a GP model.
Figure3.7illustratestheseresults.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 77
Fig.3.7 ComparisonofBNNswithGPforaregressiontaskoverthreetoydatasets.Thetoprow
isaBNNtrainedwithBayesByBackprop[76],thecentrerowistrainedwithMCdropout[70],
andthebottomaGPwithaMattern52kernelfittedwiththeGPflowpackage[120].ThetwoBNNs
consistedoftwohiddenlayersutilisingReLUactivation.Trainingdataisshownwiththedarkgrey
scatter,themeanisshowninpurple,thetruetestfunctionisshowninblue,andtheshadedregions
representing±oneandtwostd.fromthemean.Bestviewedonacomputerscreen
Analysis of the regression results shown in Fig. 3.7 shows contrasting perfor-
mance in terms of bias and variance in predictions. Models trained with Bayes
by Backprop and a factorised Gaussian approximate posterior show reasonable
predictiveresultswithinthedistributionoftrainingdata,thoughvarianceoutsidethe
regionoftrainingdataissignificantlyunderestimatedwhencomparedwiththeGP.
MCDropoutwithascaledBernoulliapproximateposteriortypicallyexhibitsgreater
varianceforoutofdistributiondata,thoughmaintainsunnecessarilyhighvariance
withinthe distributionoftrainingdata.Little tuningofhyperparameterswasdone
to these models. Better results may be achieved, particularly for MC Dropout,
withbetterselectionofhyperparameters.Alternatively,a morecompleteBayesian
approach can be used, where hyperparameters are treated as latent variables and
marginalisationisperformedoverthesevariables.
Itis worthwhilenotingthe computationalandpracticaldifficultiesencountered
withthesemethods.TheMCDropoutmethodisincrediblyversatile,inthatitwas
less sensitive to the choice of prior distribution. It also managed to fit to more

78 E.GoanandC.Fookes
complex distributions with fewer samples and training iterations. On top all this
is the significant savings in computationalresources. Given that training a model
using MC Dropout is often identical to how many existing deep networks are
trained, inference is performed in the same time as traditional vanilla networks.
It also offersno increase in the number of parameters to a network, where Bayes
byBackproprequirestwiceasmany.Thesefactorsshouldbetakenintoaccountfor
practicalscenarios.Ifthedatabeingmodelledissmooth,isinsufficientquantityand
additionaltime for inference is permitted, Bayes by Backprop may be preferable.
For large networks with complex functions, sparse data and more stringent time
requirements,MCDropoutmaybemoresuitable.
3.3.1 ConvolutionalBNNs
Whilst the MLP serves as the basis for NNs, the most prominentNN architecture
is the ConvolutionalNeuralNetwork (CNN) [121]. These networkshave excelled
atchallengingimageclassificationtasks,withpredictiveperformancefarexceeding
prior kernel based or feature engineered methods. A CNN differs from a typical
MLP through it’s application a convolution-like operator as oppose to inner
products.27Theoutputofasingleconvolutionallayercanbeexpressedas,
Φ =u(XT ∗W) (3.62)
whereu(·)isanon-linearactivationand∗representstheconvolution-likeoperation.
HeretheinputXandtheweightmatrixWarenolongerrestrictedtoeithervectorsor
matrices,andcaninsteadbemulti-dimensionalarrays.ItcanbeshownthatCNNs
can be written to have an equivalent MLP model, allowing for optimised linear
algebrapackagestobeusedfortrainingwithback-propagation[122].
Extending on the current research methods, a new type of Bayesian Convo-
lutional Neural Network (BCNN) can be developed. This is achieved here by
extending on the Bayes by Backprop method [76] to the case of models suitable
for image classification. Each weightin the convolutionallayers is assumed to be
independent,allowingforfactorisationovereachindividualparameter.
Experimentation was conducted to investigate the predictive performance of
BCNNs, and the quality of their uncertainty estimates. These networks were
configuredforclassificationoftheMNISThanddigitdataset[123].
27Emphasisisplacedon“convolutionlike”,asitisnotequivalenttothemathematicaloperation
oflinearorcircularconvolution.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 79
Sincethistaskisaclassificationtask,thelikelihoodfortheBCNNwassettoa
Softmaxfunction,
f ω (D)
softmax(f ω )= (cid:14) i (cid:10) (cid:11). (3.63)
i exp f ω (D)
j j
Theun-normalisedposteriorcanthenberepresentedas,
p(ω|D)∝p(ω)×softmax(f ω (D)). (3.64)
TheapproximateposterioristhenfoundusingBayesbyBackprop.Predictivemean
for test samples can be found using Eq. (3.60), and MC integration is used to
approximatecredibleintervals[35].
ComparisonwithavanillaCNNwasmadetoevaluatethepredictiveperformance
oftheBCNN.ForboththevanillaandBCNN,thepopularLeNetarchitecture[123]
wasused.Classification wasconductedusingthe meanoutputofthe BCNN, with
credible intervals being used to assess the models uncertainty. Overall predictive
performance for both networks on the 10,000 test images in the MNIST dataset
showedcomparativeperformance.TheBCNNshowedatestpredictionaccuracyof
98.99%,whilethe vanillanetworkshoweda slightimprovementwitha prediction
accuracy of 99.92%. Whilst the competitive predictive performance is essential,
the main benefit of the BCNN is that we yield valuable information about the
uncertainty of our predictions. Examples of difficult to classify digits are shown
in the Appendix, accompanied by plots of the mean prediction and 95% credible
intervals for each class. From these examples, we can see the large amount of
predictiveuncertaintyfor these challengingimages, which could be used to make
moreinformeddecisionsinpracticalscenarios.
This uncertainty information is invaluable for many scenarios of interest. As
statistical models are increasingly employed for complex tasks containing human
interaction,itiscrucialthatmanyofthesesystemsmakeresponsibledecisionsbased
ontheirperceivedmodeloftheworld.Forexample,NNsarelargelyusedwithinthe
developmentof autonomousvehicles. Developmentof autonomousvehicles is an
incrediblychallengingfeat,duetothehighdegreeofvariabilityinscenariosandthe
complexityrelatingto humaninteraction.Currenttechnologiesare insufficientfor
safelyenablingthistask,andasdiscussedearlier,theuseofthesetechnologieshave
been involvedin multiple deaths [24,25]. It is not possible to modelall variables
withinsuchahighlycomplexsystem.Thisaccompaniedbyimperfectmodelsand
relianceonapproximateinference,itisimportantthatourmodelscancommunicate
any uncertainty relating to decisions made. It is crucial that we acknowledgethat
in essence, our models are wrong. This is why probabilistic models are favoured
forsuchscenarios;thereisanunderlyingtheorytohelpusdealwithheterogeneity
inourdataandtoaccountforuncertaintyinducedbyvariablesnotincludedinthe
model. It is vital that models used for such complex scenarios can communicate
theiruncertaintywhenusedinsuchcomplexandhighriskscenarios.

80 E.GoanandC.Fookes
3.4 Conclusion
Throughoutthisreport,theproblemsthatarisewithoverconfidentpredictionsfrom
typicalNNsandadhocmodeldesignhavebeenillustrated.Bayesiananalysishas
been shown to provide a rich body of theory to address these challenges, though
exact computation remains analytically and computationally intractable for any
BNN of interest. In practice, approximate inference must be relied upon to yield
accurateapproximationstotheposterior.
Many of the approximate methods for inference within BNNs have revolved
aroundtheMFVBapproach.Thisprovidesatractablelowerboundtooptimisew.r.t
variationalparameters.Thesemethodsareattractiveduetotheirrelativeeaseofuse,
accuracyofpredictivemeanvaluesandacceptablenumberofinducedparameters.
Despitethis,itwasshownthroughtheliteraturesurveyandexperimentationresults
that the assumptions made within a fully factorised MFVB approach result in
over-confident predictions. It was shown that these MFVB approaches can be
extended upon to more complex models such as CNNs. Experimental results
indicate comparable predictive performance to point estimate CNNs for image
classification tasks. The Bayesian CNN was able to provide credible intervals on
thepredictions,whichwerefoundtobehighlyinformativeandintuitivemeasureof
uncertaintyfordifficulttoclassifydatapoints.
ThissurveyandtheseexperimentshighlightthecapabilitiesofBayesiananalysis
to address common challenges seen in the machine learning community. These
results also highlight how current approximate inference methods for BNNs are
insufficient and can provide inaccurate variance information. Additional research
is required to not only determine how these networks operate, but how accurate
inference can be achieved with modern large networks. Methods to scale exact
inference methods such as MCMC to large data sets would allow for a more
principled method of performing inference. MCMC offers diagnostic methods to
assessconvergenceandqualityofinference.SimilardiagnosticsforVIwouldallow
researchersandpractitionerstoevaluatethequalityoftheirassumedposterior,and
informthemwithwaysto improveonthisassumption.Achievingthese goalswill
allowustoobtainaccurateposteriorapproximations.Fromthiswewillbeableto
sufficientlydeterminewhatourmodelsknow,butalsowhattheydon’tknow.

3 BayesianNeuralNetworks:AnIntroductionandSurvey 81
Appendix
SeeFig.3.8.
| (a) | (b) | (c) |
| --- | --- | --- |
| (d) | (e) | (f) |
| (g) | (h) | (i) |
(j)
Fig.3.8 ExamplesofdifficulttoclassifyimagesfromeachclassinMNIST.Trueclassforeach
image is0–9 (a–j) arranged inalphabetical order. The bottom plot illustrates the 95% credible
intervalsforthesepredictions.Bestviewedonacomputerscreen

82 E.GoanandC.Fookes
References
1.F.Rosenblatt,Theperceptron:aprobabilisticmodelforinformationstorageandorganization
inthebrain.Psychol.Rev.65(6),386–408(1958)
2.C.Bishop,PatternRecognitionandMachineLearning(Springer,NewYork,2006)
3.D.E.Rumelhart,G.E.Hinton,R.J.Williams,Learningrepresentations byback-propagating
errors.Nature323(6088),533(1986)
4.K.-S.Oh,K.Jung,Gpuimplementationofneuralnetworks.PatternRecog.37(6),1311–1314
(2004)
5.D.C. Ciresan, U. Meier, L.M. Gambardella, J. Schmidhuber, Deep big simple neural nets
excelonhandwrittendigitrecognition.CoRR(2010)
6.A. Krizhevsky, I. Sutskever, G.E. Hinton, Imagenet classification with deep convolutional
neuralnetworks, inAdvances inNeuralInformationProcessingSystems(2012),pp.1097–
1105
7.K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale image
recognition.CoRR(2014)
8.C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke,
A.Rabinovichetal.,Goingdeeperwithconvolutions,inCVPR(2015)
9.R. Girshick, J. Donahue, T. Darrell, J. Malik, Rich feature hierarchies for accurate object
detectionandsemanticsegmentation,inProceedingsoftheIEEEConferenceonComputer
VisionandPatternRecognition(2014),pp.580–587
10.S.Ren,K.He,R.Girshick,J.Sun,Fasterr-cnn:towardsreal-timeobjectdetectionwithregion
proposalnetworks,inAdvancesinNeuralInformationProcessingSystems(2015),pp.91–99
11.J. Redmon, S. Divvala, R. Girshick, A. Farhadi, You only look once: unified, real-time
object detection, in Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition(2016),pp.779–788
12.A.Mohamed,G.E.Dahl,G.Hinton,Acousticmodelingusingdeepbeliefnetworks. IEEE
Trans.AudioSpeechLang.Process.20(1),14–22(2012)
13.G.E.Dahl,D.Yu,L.Deng,A.Acero,Context-dependent pre-traineddeepneuralnetworks
for large-vocabulary speech recognition. IEEE Trans. Audio Speech Lang. Process. 20(1),
30–42(2012)
14.G.Hinton,L.Deng,D.Yu,G.E.Dahl,A.-r.Mohamed,N.Jaitly,A.Senior,V.Vanhoucke,
P. Nguyen, T.N. Sainath et al., Deep neural networks for acoustic modeling in speech
recognition: The shared views of four research groups. IEEE Signal Process. Mag. 29(6),
82–97(2012)
15.D. Amodei, S. Ananthanarayanan, R. Anubhai, J. Bai, E. Battenberg, C. Case, J. Casper,
B.Catanzaro, Q.Cheng,G.Chen,J.Chen,J.Chen,Z.Chen,M.Chrzanowski, A.Coates,
G.Diamos,K.Ding,N.Du,E.Elsen,J.Engel,W.Fang,L.Fan,C.Fougner,L.Gao,C.Gong,
A.Hannun,T.Han,L.Johannes,B.Jiang,C.Ju,B.Jun,P.LeGresley,L.Lin,J.Liu,Y.Liu,
W. Li, X. Li, D. Ma, S. Narang, A. Ng, S. Ozair, Y. Peng, R. Prenger, S. Qian, Z. Quan,
J. Raiman, V. Rao, S. Satheesh, D. Seetapun, S. Sengupta, K. Srinet, A. Sriram, H. Tang,
L.Tang,C.Wang,J.Wang,K.Wang,Y.Wang,Z.Wang,Z.Wang,S.Wu,L.Wei,B.Xiao,
W.Xie,Y.Xie,D.Yogatama,B.Yuan,J.Zhan,Z.Zhu,Deepspeech2:end-to-endspeech
recognitioninEnglishandMandarin,inProceedingsofThe33rdInternationalConference
on Machine Learning. Proceedings of Machine Learning Research, New York, 20–22 Jun
2016,vol.48,ed.byM.F.Balcan,K.Q.Weinberger(2016),pp.173–182.
16.D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert,
L. Baker, M. Lai, A. Bolton et al., Mastering the game of go without human knowledge.
Nature550(7676),354(2017)
17.McKinsey & Company, Inc., Smartening up with artificial intelligence (ai) - what’s in it
forGermanyanditsindustrialsector?McKinsey&Company,Inc,Tech.Rep.,April2017.
Available:https://www.mckinsey.de/files/170419_mckinsey_ki_final_m.pdf