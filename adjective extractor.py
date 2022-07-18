# Importing the required libraries
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
# Function to extract the adjectives 
# Importing the required libraries
import nltk
nltk.download('averaged_perceptron_tagger')
import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
def AdjectiveExtractor(text):
    
    print('Adjectives Extracted :')
    
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'JJ' or tag == 'JJR' or tag == 'JJS': # If the word is a proper noun
                print(word)
text =  "Blogs are one of the fastest growing sections of the emerging communication mechanisms (Cohen & Krishnamurthy, 2006; Lambiotte, Ausloos, & Thelwall, 2007; Singh, Veron-Jackson, & Cullinane, 2008; Tang, Tan, & Cheng, 2009). Nowadays, many people make their opinions available on the internet and researchers have been proposing methods to automate  (Hatzivassiloglou & McKeown, 1997; Pang, Lee, & Vaithyanathan, 2002; Pang & Lee, 2008; Turney, 2002). Pang and Lee (2008) captured different definitions about these terms based on applications done in this field. Names also figure prominently, a problem noted by other researchers (Finn and Kushmerick 2003; Kennedy and Inkpen 2006). while Chmiel and Holyst [18] proposed a model of social network evolution driven by the exchange of emotional messages. Additional Web 2.0 traits include data openness and metadata, dynamic content, rich user experience and scalability tolerance (Skiba, 2006). At the same time, citizens are becoming more actively engaged in policy issues, more empowered, and more demanding in their relations with traditional institutions while political clubs, organizations, and editorials experience  (Inglehart & Welzel, 2005). The current body of work attempts to exploit the advantages of both approaches and hybrid systems are proposed, these being represented by the work of Prabowo and Thelwall (2009). Sebastiani (2002) states that machine learning based classification is practical since automatic classifiers can achieve a level of accuracy comparable to that achieved by human experts. Bloggers record the daily events in their lives and express their opinions, feelings, and emotions in a blog (Chau & Xu, 2007)."

AdjectiveExtractor(text)


def Convert(string):
	list1 = list(string.split(" "))
	return list1

# Driver code	
str1 = "powerful essential concept recent learnable huge manual critical context generalizable unseen propose lightweight neural generate static dynamic adapt sensitive Extensive better unseen single stronger domain Recent striking zero-shot potential open-world visual key visual traditional supervised minimize Such closed-set visual pre-defined unscalable new unseen vision-language parameterized e.g. different prompt type class-specific filling token real vision-language natural open-set visual effective transferable powerful vision-language investigate potential web-scale high enormous fine-tuning entire deep impractical well-learned safer prompt context meaningful effective prompt time-consuming optimal prompt recent pre-trained vision-language prompt learnable differentiable nature neural huge intensively-tuned manual wide critical context generalizable unseen cathedral significant new unseen ’ context generalizable vital broader static specific contrary manually-designed zero-shot generalizable weak novel conditional key input effective conditional lightweight neural generate input-conditional learnable approach overview paradigm analogous  instanceconditional generalizable robust specific comprehensive visual base-to-new new zero-shot overall Table significant unseen different clear Table instance-conditional transferable potential larger stronger domain dynamic summary insights effectiveness simple various approach future Related new different typical key third related early extra hand-crafted  neural pre-trained  frequency-based cross-modality common metric  multi-label n-gram visual transferable  Recent larger neural recent contrastive  web-scale representative neural network-based contrastive remarkable zero-shot Similar orthogonal efficient pre-trained vision-language pre-trained useful  pre-trained “ fill-in-the-blank token positive negative key underlined format familiar affordable-sized text optimal highest propose gradient-based vocabulary best greatest Meta-Net  v1 + token learnable neural input-conditional continuous main prompt continuous optimized objective comprehensive prompt nascent  top earliest continuous pre-trained weak simple conditional novel Zero-Shot relevant similar i.e. novel novel “ seen-class common semantic auxiliary large vision-language different first brief present technical rationale applicable broader CLIP-like enormous sufficient used follow parameter-efficient  insights static demonstrate conditional new cross-dataset future conditional efficient cross-dataset instance-conditional static bigger larger-scale heterogeneous mixed different in-kind  Food-101–mining discriminative random  empirical object  simple contrastive visual   large-scale hierarchical  visual textual  Pre-training deep  worth  Zero-shot textual  generative visual incremental bayesian object  deep visual-semantic  ¨ Modern outperform    pretrained few-shot   Self-supervised visual text  unsupervised visual  Deep residual novel deep  Data-efficient contrastive predictive  many critical  adversarial  Fine-grained zero-shot attribute-based  visual vision-language noisy text know visual large  efficient   object fine-grained  deep zero-shot convolutional neural textual  parameter-efficient  visual n-grams  continuous  contrastive language-image pre-training  systematic natural visual flower large neural"
print(Convert(str1))


def Convert(string):
	list2 = list(string.split(" "))
	return list2

# Driver code	
str2 = "label colorization temporal array inconsistent difficult unified study important unsupervised higher previous fair recent video-based best strong usage video-based Additional similar current inform future temporal initial subsequent Popular best large semantic label top semantic human additional online large-scale pixel-level difficult recent self-supervised temporal feature pre-trained different desired label Popular self-supervised frame-toframe ] propagation similar previous pre-trained label recent previous observed related late stride ImageNet understandable label pretext mundane nearest hand-crafted much underlying ImageNet-pretrained recent ImageNet-pretrained due baseline late justified spend last unrelated technical similar strongest possible essential clear-eyed suboptimal inconsistent difficult understand much spirit comparative critical metric [ self-supervised [ recent self-supervised several feature label summarizes several key main unsupervised higher previous Next possible improve video-based hybrid ] video-based self-supervised still-image-based promising future Next fair comparison recent top-performing light several recent similar strong different similar best Generic self-supervised adopt inconsistent hand-crafted important difficult start PyTorch-like ] full aspect Further ] come overall spatial Top highest soft label sum top respective normalized “ per-frame w.r.t overall softmax top normalized consistent determine sharp propagation affect final • overall top • • examine feature self-supervised temporal describes supervised self-supervised ] recent video prominent last softmax top-k top-k pose observed average label 50-80 generic previous top unsupervised sense well-known ResNet18 Supervised trained image-level strong baseline temporal useful early much finer scale receptive semantic Self-supervised recent self-supervised representative self-supervised first straightforward effective many auxiliary second recent contrastive top downstream understanding Video-Based Self-supervised b similar first grayscale Numerous ] temporal supervision random frame second spatial differentiable supervisory longer method per-frame spatial ] higher-level task temporal similar auxiliary spatial grid u coordinate method substantial cross-entropy decoded equivalent orthogonal concentration per-frame “ test-time sensitive ] follow-up Self-supervised ] colorization-based temporal similar grayscale effective CorrFlow cross-entropy effective sophisticated memory-based spatial per-frame softmax square spatial context separate ] temporal primary grid shortcut single patch subsequent probabilistic backward several many longer stronger object-level matrix different full individual applied overall single context test consecutive spatial ] image-based image-level high learned top typical imagebased learned comparable top self-supervised overall spatial examine Overall per-frame key context “ per-frame separate context overall single context softmax applied overall per-frame important softmax top overall per-frame per-frame best fair overall best per-frame subsequent best context n. lead meaningful overall ” “ per-frame intermediate Previous different little ] context first first fifth single frame main provide similar frame additional previous context first context sensitive overall low best “ per-frame sensitive different improve larger different different k better spatial next Spatial several previous form particular described important dependent mask trainable per-frame object object simpler fixed context particular reduced-size sophisticated longer report best difficult second image-based videobased semantic Table compare recent spatial additional semantic spatial additional lower recent best additional similar studied strong fair initial interesting self-supervised ] semantic large pixel-level ] human human joint joint previous different test correct metric ] threshold Table comparative recent ImageNet stride fine-tuning third previous second Self-supervised image-based comparable top video-based similar small different substantial recent spatial context categorize different previous [ works systematic temporal particular Much several methodological best better future temporal critical certain well-optimized video-based different recent high-performing similar welltuned much particular tracking-based global potential correspon4This self-supervised standard supervised supervised separate visible fastest many available different social emotional dynamic rich traditional political current hybrid practical automatic comparable human daily express sentiment complete due semi-supervised domain independent micro-blogging emotional large-scaleable online first large"
print(Convert(str2))

list1 = list(str1.split(" "))
list2 = list(str2.split(" "))




list3 = set(list1)&set(list2)

list4 = sorted(list3, key = lambda k : list1.index(k))
print(list4)


list5=['essential', 'recent', 'critical', 'sensitive', 'better', 'single', 'stronger', 'potential', 'key', 'supervised', 'different', 'effective', 'high', 'meaningful', 'differentiable', 'overall', 'larger', 'future', 'typical', 'third', 'related', 'early', 'metric', 'contrastive', 'representative', 'orthogonal', 'useful', 'highest', 'best', 'main', 'top', 'similar', 'semantic', 'auxiliary', 'large', 'first', 'technical', 'random', 'object', 'unsupervised', 'many', 'systematic', ]
score=[0.1,0,0,-0.1,0.2,0,0.3,0,0,0.1,0.5,0,0.6,0,0.1,0.3,0,-0.3,0,0,0,0,0,0,0,0.7,0.4,0.9,0.2,0,0,0,0,0.2,0,0,0,0,0,0.1,0,0.2,0.4,0.6,0.1,-0.6,-0.4,0.2,-0.4]

score_dictionary = dict(zip(list5, score))