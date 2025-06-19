import React, { useState, useEffect } from 'react';
import { 
  Clock, 
  BookOpen, 
  Target, 
  TrendingUp, 
  CheckCircle2, 
  Circle, 
  Star, 
  ExternalLink, 
  Calendar, 
  DollarSign,
  Users,
  Award,
  Brain,
  Code,
  Zap,
  Globe,
  ChevronDown,
  ChevronRight,
  Play,
  Pause,
  RotateCcw,
  Book,
  Video,
  FileText,
  Monitor,
  Lightbulb
} from 'lucide-react';

const AiEngineerRoadmap = () => {
  const [selectedPhase, setSelectedPhase] = useState(0);
  const [completedMilestones, setCompletedMilestones] = useState(new Set());
  const [completedSkills, setCompletedSkills] = useState(new Set());
  const [completedResources, setCompletedResources] = useState(new Set());
  const [isTimerActive, setIsTimerActive] = useState(false);
  const [studyTime, setStudyTime] = useState(0);

  // Enhanced roadmap data with integrated phase structure
  const roadmapData = {
    metadata: {
      title: "Software Engineer to AI Engineer Transition Roadmap",
      version: "2.0",
      duration: "6-12 months",
      timeCommitment: "15-20 hours/week",
      difficulty: "Intermediate to Advanced",
      totalPhases: 5
    },
    phases: [
      {
        id: 0,
        title: "Foundation Building",
        duration: "6-8 weeks",
        priority: "Critical",
        description: "Build essential mathematical and Python foundations for AI",
        color: "bg-blue-500",
        icon: "🔢",
        totalHours: "100-120 hours",
        keySkills: [
          {
            id: "math-foundations",
            title: "Mathematical Foundations",
            category: "Mathematics",
            importance: "Critical",
            timeRequired: "60-70 hours",
            topics: [
              "Linear Algebra (vectors, matrices, eigenvalues)",
              "Calculus (derivatives, chain rule, optimization)",
              "Statistics & Probability (distributions, Bayes theorem)",
              "Discrete Mathematics basics"
            ],
            completed: false,
            resources: [
              {
                id: "linear-algebra-course",
                title: "Linear Algebra - Mathematics for Machine Learning",
                provider: "Coursera (Imperial College)",
                type: "course",
                cost: "Free audit / $49 certificate",
                rating: 4.6,
                duration: "4 weeks",
                url: "https://www.coursera.org/learn/linear-algebra-machine-learning",
                completed: false
              },
              {
                id: "khan-linear-algebra",
                title: "Khan Academy Linear Algebra",
                provider: "Khan Academy",
                type: "tutorial",
                cost: "Free",
                rating: 4.8,
                duration: "Self-paced",
                url: "https://www.khanacademy.org/math/linear-algebra",
                completed: false
              },
              {
                id: "calculus-course",
                title: "Multivariate Calculus for ML",
                provider: "Coursera (Imperial College)",
                type: "course",
                cost: "Free audit / $49 certificate",
                rating: 4.7,
                duration: "4 weeks",
                url: "https://www.coursera.org/learn/multivariate-calculus-machine-learning",
                completed: false
              },
              {
                id: "statistics-course",
                title: "Statistics and Probability in Data Science",
                provider: "edX (UC San Diego)",
                type: "course",
                cost: "Free audit / $99 certificate",
                rating: 4.5,
                duration: "10 weeks",
                url: "https://www.edx.org/course/statistics-and-probability-in-data-science-using-python",
                completed: false
              }
            ]
          },
          {
            id: "python-ai",
            title: "Python for AI & Data Science",
            category: "Programming",
            importance: "Critical",
            timeRequired: "40-50 hours",
            topics: [
              "Advanced Python concepts (OOP, functional programming)",
              "NumPy for numerical computing",
              "Pandas for data manipulation",
              "Matplotlib & Seaborn for visualization",
              "Jupyter notebooks and development environment"
            ],
            completed: false,
            resources: [
              {
                id: "python-everybody",
                title: "Python for Everybody Specialization",
                provider: "Coursera (University of Michigan)",
                type: "course",
                cost: "Free audit / $49/month",
                rating: 4.8,
                duration: "8 months",
                url: "https://www.coursera.org/specializations/python",
                completed: false
              },
              {
                id: "datacamp-python",
                title: "Python Programmer Track",
                provider: "DataCamp",
                type: "interactive",
                cost: "$35/month",
                rating: 4.6,
                duration: "36 hours",
                url: "https://www.datacamp.com/tracks/python-programmer",
                completed: false
              },
              {
                id: "python-data-analysis",
                title: "Introduction to Data Science in Python",
                provider: "Coursera (University of Michigan)",
                type: "course",
                cost: "Free audit / $49 certificate",
                rating: 4.5,
                duration: "4 weeks",
                url: "https://www.coursera.org/learn/python-data-analysis",
                completed: false
              },
              {
                id: "numpy-pandas-tutorial",
                title: "NumPy and Pandas Masterclass",
                provider: "Real Python",
                type: "tutorial",
                cost: "Free",
                rating: 4.8,
                duration: "10 hours",
                url: "https://realpython.com/numpy-tutorial/",
                completed: false
              }
            ]
          }
        ],
        milestones: [
          { week: 2, text: "Complete linear algebra fundamentals and matrix operations", completed: false },
          { week: 4, text: "Master NumPy operations and Pandas data manipulation", completed: false },
          { week: 6, text: "Understand calculus concepts for ML (gradients, optimization)", completed: false },
          { week: 8, text: "Build comprehensive data analysis project using Python stack", completed: false }
        ],
        projects: [
          {
            title: "Mathematical Foundations Portfolio",
            description: "Implement linear algebra operations from scratch and solve optimization problems",
            deliverables: ["Matrix operations library", "Gradient descent implementation", "Statistical analysis notebook"]
          },
          {
            title: "Python Data Analysis Project",
            description: "Analyze a real-world dataset using NumPy, Pandas, and visualization libraries",
            deliverables: ["Data cleaning pipeline", "Exploratory data analysis", "Visualization dashboard"]
          }
        ]
      },
      {
        id: 1,
        title: "Machine Learning Fundamentals",
        duration: "8-10 weeks",
        priority: "Critical",
        description: "Master core machine learning algorithms and implementation techniques",
        color: "bg-green-500",
        icon: "🤖",
        totalHours: "80-100 hours",
        keySkills: [
          {
            id: "supervised-learning",
            title: "Supervised Learning Algorithms",
            category: "Machine Learning",
            importance: "Critical",
            timeRequired: "35-40 hours",
            topics: [
              "Linear and Logistic Regression",
              "Decision Trees and Random Forests",
              "Support Vector Machines (SVM)",
              "K-Nearest Neighbors (KNN)",
              "Naive Bayes Classification",
              "Model evaluation and cross-validation"
            ],
            completed: false,
            resources: [
              {
                id: "andrew-ng-ml",
                title: "Machine Learning Course (Andrew Ng)",
                provider: "Coursera (Stanford)",
                type: "course",
                cost: "Free audit / $79 certificate",
                rating: 4.9,
                duration: "11 weeks",
                url: "https://www.coursera.org/learn/machine-learning",
                completed: false
              },
              {
                id: "hands-on-ml-book",
                title: "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
                provider: "Aurélien Géron",
                type: "book",
                cost: "$45",
                rating: 4.7,
                duration: "Self-paced",
                url: "https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646",
                completed: false
              },
              {
                id: "mit-ml-course",
                title: "Introduction to Machine Learning with Python",
                provider: "edX (MIT)",
                type: "course",
                cost: "Free audit / $99 certificate",
                rating: 4.6,
                duration: "9 weeks",
                url: "https://www.edx.org/course/introduction-to-machine-learning-with-python",
                completed: false
              }
            ]
          },
          {
            id: "unsupervised-learning",
            title: "Unsupervised Learning & Feature Engineering",
            category: "Machine Learning",
            importance: "High",
            timeRequired: "25-30 hours",
            topics: [
              "K-means and Hierarchical Clustering",
              "Principal Component Analysis (PCA)",
              "t-SNE and UMAP for dimensionality reduction",
              "Feature selection and engineering",
              "Association rule learning"
            ],
            completed: false,
            resources: [
              {
                id: "unsupervised-datacamp",
                title: "Unsupervised Learning in Python",
                provider: "DataCamp",
                type: "course",
                cost: "$35/month",
                rating: 4.5,
                duration: "4 hours",
                url: "https://www.datacamp.com/courses/unsupervised-learning-in-python",
                completed: false
              },
              {
                id: "feature-engineering-course",
                title: "Feature Engineering for Machine Learning",
                provider: "Coursera",
                type: "course",
                cost: "Free audit / $49 certificate",
                rating: 4.4,
                duration: "5 weeks",
                url: "https://www.coursera.org/learn/feature-engineering",
                completed: false
              }
            ]
          },
          {
            id: "ml-tools",
            title: "ML Tools & Libraries",
            category: "Tools",
            importance: "Critical",
            timeRequired: "20-30 hours",
            topics: [
              "Scikit-learn ecosystem mastery",
              "Model selection and hyperparameter tuning",
              "Pipeline construction and automation",
              "Model evaluation metrics and validation",
              "Jupyter notebook best practices"
            ],
            completed: false,
            resources: [
              {
                id: "scikit-learn-tutorial",
                title: "Scikit-learn User Guide",
                provider: "Scikit-learn.org",
                type: "tutorial",
                cost: "Free",
                rating: 4.8,
                duration: "15 hours",
                url: "https://scikit-learn.org/stable/user_guide.html",
                completed: false
              },
              {
                id: "ml-python-ibm",
                title: "Machine Learning with Python",
                provider: "IBM (Coursera)",
                type: "course",
                cost: "Free audit / $49 certificate",
                rating: 4.6,
                duration: "5 weeks",
                url: "https://www.coursera.org/learn/machine-learning-with-python",
                completed: false
              }
            ]
          }
        ],
        milestones: [
          { week: 2, text: "Implement basic ML algorithms from scratch (linear regression, decision tree)", completed: false },
          { week: 4, text: "Master scikit-learn workflow and achieve >85% accuracy on standard datasets", completed: false },
          { week: 6, text: "Build and evaluate classification and regression models", completed: false },
          { week: 8, text: "Complete feature engineering and model optimization project", completed: false },
          { week: 10, text: "Deploy end-to-end ML project with proper evaluation metrics", completed: false }
        ],
        projects: [
          {
            title: "ML Algorithm Implementation",
            description: "Implement core ML algorithms from scratch using only NumPy",
            deliverables: ["Linear regression implementation", "Decision tree classifier", "K-means clustering"]
          },
          {
            title: "Predictive Analytics Project",
            description: "Build a complete ML pipeline for a real-world prediction task",
            deliverables: ["Data preprocessing pipeline", "Model comparison study", "Production-ready model with API"]
          }
        ]
      },
      {
        id: 2,
        title: "Deep Learning & Neural Networks",
        duration: "10-12 weeks",
        priority: "Critical",
        description: "Master deep learning architectures and modern AI frameworks",
        color: "bg-purple-500",
        icon: "🧠",
        totalHours: "120-140 hours",
        keySkills: [
          {
            id: "neural-networks",
            title: "Neural Network Fundamentals",
            category: "Deep Learning",
            importance: "Critical",
            timeRequired: "40-45 hours",
            topics: [
              "Perceptron and Multi-layer Perceptrons",
              "Backpropagation algorithm and gradient descent",
              "Activation functions and their properties",
              "Loss functions and optimization techniques",
              "Regularization (dropout, batch normalization, L1/L2)"
            ],
            completed: false,
            resources: [
              {
                id: "deep-learning-specialization",
                title: "Deep Learning Specialization",
                provider: "Coursera (DeepLearning.AI)",
                type: "course",
                cost: "$49/month",
                rating: 4.9,
                duration: "4 months",
                url: "https://www.coursera.org/specializations/deep-learning",
                completed: false
              },
              {
                id: "cs231n-stanford",
                title: "CS231n: Convolutional Neural Networks",
                provider: "Stanford (YouTube)",
                type: "video",
                cost: "Free",
                rating: 4.9,
                duration: "20 hours",
                url: "https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv",
                completed: false
              },
              {
                id: "deep-learning-book",
                title: "Deep Learning Book",
                provider: "Goodfellow, Bengio, Courville",
                type: "book",
                cost: "Free online / $65 paperback",
                rating: 4.8,
                duration: "Self-paced",
                url: "https://www.deeplearningbook.org/",
                completed: false
              }
            ]
          },
          {
            id: "cnn-architecture",
            title: "Convolutional Neural Networks",
            category: "Computer Vision",
            importance: "Critical",
            timeRequired: "30-35 hours",
            topics: [
              "Convolution operation and feature maps",
              "Pooling layers and architectural patterns",
              "Classic architectures (LeNet, AlexNet, VGG, ResNet)",
              "Transfer learning and fine-tuning",
              "Image classification and object detection"
            ],
            completed: false,
            resources: [
              {
                id: "cnn-course",
                title: "Convolutional Neural Networks",
                provider: "Coursera (DeepLearning.AI)",
                type: "course",
                cost: "$49/month",
                rating: 4.9,
                duration: "4 weeks",
                url: "https://www.coursera.org/learn/convolutional-neural-networks",
                completed: false
              },
              {
                id: "fastai-course",
                title: "Practical Deep Learning for Coders",
                provider: "Fast.ai",
                type: "course",
                cost: "Free",
                rating: 4.8,
                duration: "7 weeks",
                url: "https://course.fast.ai/",
                completed: false
              }
            ]
          },
          {
            id: "rnn-nlp",
            title: "Recurrent Networks & NLP",
            category: "Natural Language Processing",
            importance: "High",
            timeRequired: "30-35 hours",
            topics: [
              "Recurrent Neural Networks (RNN) fundamentals",
              "LSTM and GRU architectures",
              "Sequence-to-sequence models",
              "Attention mechanisms and Transformers",
              "Text preprocessing and embedding techniques"
            ],
            completed: false,
            resources: [
              {
                id: "sequence-models",
                title: "Sequence Models",
                provider: "Coursera (DeepLearning.AI)",
                type: "course",
                cost: "$49/month",
                rating: 4.8,
                duration: "3 weeks",
                url: "https://www.coursera.org/learn/nlp-sequence-models",
                completed: false
              },
              {
                id: "cs224n-stanford",
                title: "CS224N: Natural Language Processing with Deep Learning",
                provider: "Stanford",
                type: "video",
                cost: "Free",
                rating: 4.9,
                duration: "20 hours",
                url: "http://web.stanford.edu/class/cs224n/",
                completed: false
              }
            ]
          },
          {
            id: "dl-frameworks",
            title: "Deep Learning Frameworks",
            category: "Tools",
            importance: "Critical",
            timeRequired: "25-30 hours",
            topics: [
              "TensorFlow 2.x and Keras high-level API",
              "PyTorch fundamentals and training loops",
              "Model deployment and serving",
              "TensorBoard and experiment tracking",
              "GPU optimization and distributed training"
            ],
            completed: false,
            resources: [
              {
                id: "tensorflow-developer",
                title: "TensorFlow Developer Certificate",
                provider: "Coursera (DeepLearning.AI)",
                type: "course",
                cost: "$49/month",
                rating: 4.7,
                duration: "4 months",
                url: "https://www.coursera.org/professional-certificates/tensorflow-in-practice",
                completed: false
              },
              {
                id: "pytorch-tutorial",
                title: "PyTorch for Deep Learning",
                provider: "Udacity",
                type: "course",
                cost: "Free",
                rating: 4.6,
                duration: "4 weeks",
                url: "https://www.udacity.com/course/deep-learning-pytorch--ud188",
                completed: false
              },
              {
                id: "tensorflow-official",
                title: "TensorFlow Official Tutorials",
                provider: "Google",
                type: "tutorial",
                cost: "Free",
                rating: 4.8,
                duration: "20 hours",
                url: "https://www.tensorflow.org/tutorials",
                completed: false
              }
            ]
          }
        ],
        milestones: [
          { week: 3, text: "Build and train neural network from scratch using NumPy", completed: false },
          { week: 6, text: "Achieve >90% accuracy on CIFAR-10 using CNN", completed: false },
          { week: 9, text: "Implement RNN for sentiment analysis with >85% accuracy", completed: false },
          { week: 12, text: "Deploy production-ready deep learning model with monitoring", completed: false }
        ],
        projects: [
          {
            title: "Computer Vision Application",
            description: "Build an end-to-end image classification or object detection system",
            deliverables: ["Custom CNN architecture", "Transfer learning implementation", "Web deployment with API"]
          },
          {
            title: "NLP Text Analysis System",
            description: "Create a comprehensive text analysis tool using RNNs/Transformers",
            deliverables: ["Text preprocessing pipeline", "Multi-class classification model", "Real-time inference API"]
          }
        ]
      },
      {
        id: 3,
        title: "Advanced AI & Specialization",
        duration: "8-10 weeks",
        priority: "High",
        description: "Master cutting-edge AI techniques and production deployment",
        color: "bg-orange-500",
        icon: "🚀",
        totalHours: "100-120 hours",
        keySkills: [
          {
            id: "transformers-llms",
            title: "Transformers & Large Language Models",
            category: "Advanced NLP",
            importance: "Critical",
            timeRequired: "35-40 hours",
            topics: [
              "Attention mechanisms and self-attention",
              "Transformer architecture (encoder-decoder)",
              "Pre-trained models (BERT, GPT, T5, LLaMA)",
              "Fine-tuning and transfer learning strategies",
              "Prompt engineering and in-context learning"
            ],
            completed: false,
            resources: [
              {
                id: "huggingface-course",
                title: "Hugging Face NLP Course",
                provider: "Hugging Face",
                type: "course",
                cost: "Free",
                rating: 4.8,
                duration: "8 weeks",
                url: "https://huggingface.co/course/chapter1/1",
                completed: false
              },
              {
                id: "attention-paper",
                title: "Attention Is All You Need (Original Paper)",
                provider: "arXiv",
                type: "paper",
                cost: "Free",
                rating: 4.9,
                duration: "4 hours",
                url: "https://arxiv.org/abs/1706.03762",
                completed: false
              },
              {
                id: "generative-ai-llms",
                title: "Generative AI with Large Language Models",
                provider: "Coursera (DeepLearning.AI)",
                type: "course",
                cost: "$49/month",
                rating: 4.7,
                duration: "3 weeks",
                url: "https://www.coursera.org/learn/generative-ai-with-llms",
                completed: false
              }
            ]
          },
          {
            id: "generative-ai",
            title: "Generative AI & Advanced Architectures",
            category: "Generative Models",
            importance: "High",
            timeRequired: "30-35 hours",
            topics: [
              "Generative Adversarial Networks (GANs)",
              "Variational Autoencoders (VAEs)",
              "Diffusion models and stable diffusion",
              "Neural style transfer and image generation",
              "Multimodal AI and vision-language models"
            ],
            completed: false,
            resources: [
              {
                id: "gans-course",
                title: "Generative Adversarial Networks (GANs) Specialization",
                provider: "Coursera (DeepLearning.AI)",
                type: "course",
                cost: "$49/month",
                rating: 4.7,
                duration: "3 months",
                url: "https://www.coursera.org/specializations/generative-adversarial-networks-gans",
                completed: false
              },
              {
                id: "gans-book",
                title: "GANs in Action",
                provider: "Manning Publications",
                type: "book",
                cost: "$45",
                rating: 4.5,
                duration: "Self-paced",
                url: "https://www.manning.com/books/gans-in-action",
                completed: false
              }
            ]
          },
          {
            id: "mlops-production",
            title: "MLOps & Production AI Systems",
            category: "MLOps",
            importance: "Critical",
            timeRequired: "35-45 hours",
            topics: [
              "ML pipeline design and automation",
              "Model versioning and experiment tracking",
              "Continuous integration/deployment for ML",
              "Model monitoring and performance tracking",
              "A/B testing for ML models and feature flags"
            ],
            completed: false,
            resources: [
              {
                id: "mlops-specialization",
                title: "Machine Learning Engineering for Production (MLOps)",
                provider: "Coursera (DeepLearning.AI)",
                type: "course",
                cost: "$49/month",
                rating: 4.6,
                duration: "4 months",
                url: "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops",
                completed: false
              },
              {
                id: "mlflow-tutorial",
                title: "MLflow Complete Tutorial",
                provider: "MLflow.org",
                type: "tutorial",
                cost: "Free",
                rating: 4.7,
                duration: "10 hours",
                url: "https://mlflow.org/docs/latest/tutorials-and-examples/index.html",
                completed: false
              },
              {
                id: "ml-design-book",
                title: "Designing Machine Learning Systems",
                provider: "Chip Huyen",
                type: "book",
                cost: "$55",
                rating: 4.8,
                duration: "Self-paced",
                url: "https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969",
                completed: false
              }
            ]
          }
        ],
        milestones: [
          { week: 2, text: "Implement attention mechanism and basic transformer from scratch", completed: false },
          { week: 4, text: "Fine-tune BERT model and achieve SOTA results on NLP task", completed: false },
          { week: 6, text: "Deploy ML model to cloud with proper monitoring and logging", completed: false },
          { week: 8, text: "Build complete MLOps pipeline with automated training and deployment", completed: false },
          { week: 10, text: "Create production-ready AI system with scalability and monitoring", completed: false }
        ],
        projects: [
          {
            title: "Advanced NLP Application",
            description: "Build a sophisticated NLP application using transformers",
            deliverables: ["Fine-tuned transformer model", "Multi-task learning setup", "Production API with caching"]
          },
          {
            title: "MLOps Pipeline Implementation",
            description: "Create a complete MLOps pipeline for model lifecycle management",
            deliverables: ["Automated training pipeline", "Model monitoring dashboard", "A/B testing framework"]
          }
        ]
      },
      {
        id: 4,
        title: "Professional Development & Portfolio",
        duration: "4-6 weeks",
        priority: "Critical",
        description: "Build professional portfolio and prepare for AI Engineer interviews",
        color: "bg-red-500",
        icon: "💼",
        totalHours: "80-100 hours",
        keySkills: [
          {
            id: "portfolio-development",
            title: "Technical Portfolio & Communication",
            category: "Professional Skills",
            importance: "Critical",
            timeRequired: "40-50 hours",
            topics: [
              "GitHub portfolio optimization and documentation",
              "Technical blog writing and case studies",
              "Open source contributions and community involvement",
              "Code quality, testing, and best practices",
              "Presentation skills and technical communication"
            ],
            completed: false,
            resources: [
              {
                id: "portfolio-guide",
                title: "Building a World-Class Data Science Portfolio",
                provider: "Towards Data Science",
                type: "tutorial",
                cost: "Free",
                rating: 4.7,
                duration: "5 hours",
                url: "https://towardsdatascience.com/how-to-build-a-data-science-portfolio-5f566517c79c",
                completed: false
              },
              {
                id: "technical-writing",
                title: "Technical Writing Courses",
                provider: "Google Developers",
                type: "course",
                cost: "Free",
                rating: 4.6,
                duration: "Self-paced",
                url: "https://developers.google.com/tech-writing",
                completed: false
              },
              {
                id: "kaggle-competitions",
                title: "Kaggle Learn & Competitions",
                provider: "Kaggle",
                type: "platform",
                cost: "Free",
                rating: 4.8,
                duration: "Ongoing",
                url: "https://www.kaggle.com/learn",
                completed: false
              }
            ]
          },
          {
            id: "interview-preparation",
            title: "AI Engineer Interview Mastery",
            category: "Interview Skills",
            importance: "Critical",
            timeRequired: "30-40 hours",
            topics: [
              "ML system design and architecture interviews",
              "Coding challenges and algorithm implementation",
              "Model debugging and optimization scenarios",
              "Case study analysis and problem-solving",
              "Behavioral interviews and leadership principles"
            ],
            completed: false,
            resources: [
              {
                id: "ml-system-design",
                title: "Machine Learning System Design Interview",
                provider: "educative.io",
                type: "course",
                cost: "$79/month",
                rating: 4.4,
                duration: "Self-paced",
                url: "https://www.educative.io/path/machine-learning-system-design-interview",
                completed: false
              },
              {
                id: "leetcode-premium",
                title: "LeetCode Premium (ML/AI Questions)",
                provider: "LeetCode",
                type: "platform",
                cost: "$35/month",
                rating: 4.5,
                duration: "Ongoing",
                url: "https://leetcode.com/problemset/all/",
                completed: false
              },
              {
                id: "cracking-ml-interview",
                title: "Cracking the Machine Learning Interview",
                provider: "Various Authors",
                type: "book",
                cost: "$40",
                rating: 4.6,
                duration: "Self-paced",
                url: "https://www.amazon.com/dp/B08R1G8ZQV",
                completed: false
              }
            ]
          },
          {
            id: "industry-knowledge",
            title: "Industry Knowledge & Networking",
            category: "Professional Network",
            importance: "High",
            timeRequired: "15-20 hours",
            topics: [
              "AI industry trends and emerging technologies",
              "Professional networking and community building",
              "Conference participation and knowledge sharing",
              "Mentorship and continuous learning strategies",
              "Career positioning and personal branding"
            ],
            completed: false,
            resources: [
              {
                id: "ai-conferences",
                title: "Major AI Conferences (NeurIPS, ICML, ICLR)",
                provider: "Various",
                type: "conference",
                cost: "$500-2000",
                rating: 4.9,
                duration: "3-5 days each",
                url: "https://neurips.cc/",
                completed: false
              },
              {
                id: "ai-newsletters",
                title: "The Batch - AI Newsletter",
                provider: "DeepLearning.AI",
                type: "newsletter",
                cost: "Free",
                rating: 4.8,
                duration: "Weekly",
                url: "https://www.deeplearning.ai/the-batch/",
                completed: false
              }
            ]
          }
        ],
        milestones: [
          { week: 2, text: "Complete and deploy first comprehensive portfolio project", completed: false },
          { week: 4, text: "Publish technical blog post or contribute to open source project", completed: false },
          { week: 6, text: "Portfolio review-ready with 3+ high-quality AI projects", completed: false }
        ],
        projects: [
          {
            title: "Capstone AI Project",
            description: "Build a comprehensive AI application demonstrating end-to-end expertise",
            deliverables: ["Full-stack AI application", "Technical documentation", "Deployment on cloud platform"]
          },
          {
            title: "Technical Content Creation",
            description: "Create educational content demonstrating your AI expertise",
            deliverables: ["Technical blog series", "Video tutorials or demos", "Open source contribution"]
          },
          {
            title: "Interview Portfolio",
            description: "Prepare materials and practice for AI Engineer interviews",
            deliverables: ["System design portfolio", "Coding challenge solutions", "Mock interview recordings"]
          }
        ]
      }
    ],
    careerInfo: {
      salaryRanges: {
        entryLevel: "$80,000 - $120,000",
        experienced: "$120,000 - $200,000+",
        senior: "$200,000 - $400,000+"
      },
      jobMarket: {
        demand: "High",
        growth: "15% annually",
        remoteWork: "70% of positions"
      },
      targetCompanies: {
        bigTech: ["Google", "Microsoft", "Amazon", "Meta", "Apple"],
        aiFocused: ["OpenAI", "Anthropic", "DeepMind", "Hugging Face", "Scale AI"],
        startups: ["Numerous AI startups across industries"]
      }
    }
  };

  // Timer functionality
  useEffect(() => {
    let interval;
    if (isTimerActive) {
      interval = setInterval(() => {
        setStudyTime(time => time + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isTimerActive]);

  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const toggleMilestone = (phaseId, milestoneIndex) => {
    const key = `${phaseId}-${milestoneIndex}`;
    const newCompleted = new Set(completedMilestones);
    if (newCompleted.has(key)) {
      newCompleted.delete(key);
    } else {
      newCompleted.add(key);
    }
    setCompletedMilestones(newCompleted);
  };

  const toggleSkill = (skillId) => {
    const newCompleted = new Set(completedSkills);
    if (newCompleted.has(skillId)) {
      newCompleted.delete(skillId);
    } else {
      newCompleted.add(skillId);
    }
    setCompletedSkills(newCompleted);
  };

  const toggleResource = (resourceId) => {
    const newCompleted = new Set(completedResources);
    if (newCompleted.has(resourceId)) {
      newCompleted.delete(resourceId);
    } else {
      newCompleted.add(resourceId);
    }
    setCompletedResources(newCompleted);
  };

  const getPhaseProgress = (phase) => {
    const totalItems = phase.milestones.length + 
      phase.keySkills.reduce((acc, skill) => acc + skill.resources.length, 0) + 
      phase.keySkills.length;
    
    const completedItems = 
      phase.milestones.filter((_, index) => completedMilestones.has(`${phase.id}-${index}`)).length +
      phase.keySkills.filter(skill => completedSkills.has(skill.id)).length +
      phase.keySkills.reduce((acc, skill) => 
        acc + skill.resources.filter(resource => completedResources.has(resource.id)).length, 0
      );
    
    return Math.round((completedItems / totalItems) * 100);
  };

  const getOverallProgress = () => {
    const totalProgress = roadmapData.phases.reduce((acc, phase) => acc + getPhaseProgress(phase), 0);
    return Math.round(totalProgress / roadmapData.phases.length);
  };

  const ProgressRing = ({ progress, size = 60 }) => {
    const radius = (size - 4) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDasharray = `${(progress / 100) * circumference} ${circumference}`;

    return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="transform -rotate-90" width={size} height={size}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth="2"
            fill="transparent"
            className="text-gray-200"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth="2"
            fill="transparent"
            strokeDasharray={strokeDasharray}
            className="text-blue-500 transition-all duration-300"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`font-semibold ${size > 50 ? 'text-sm' : 'text-xs'}`}>{progress}%</span>
        </div>
      </div>
    );
  };

  const PhaseCard = ({ phase, isSelected, onClick }) => {
    const progress = getPhaseProgress(phase);
    
    return (
      <div
        className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
          isSelected 
            ? 'border-blue-500 bg-blue-50 shadow-lg' 
            : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
        }`}
        onClick={onClick}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-4 h-4 rounded-full ${phase.color}`}></div>
            <span className="text-lg">{phase.icon}</span>
          </div>
          <ProgressRing progress={progress} size={40} />
        </div>
        
        <h3 className="font-semibold text-gray-900 mb-2">{phase.title}</h3>
        <p className="text-sm text-gray-600 mb-3 line-clamp-2">{phase.description}</p>
        
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            <span>{phase.duration}</span>
          </div>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
            phase.priority === 'Critical' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
          }`}>
            {phase.priority}
          </span>
        </div>
      </div>
    );
  };

  const getResourceIcon = (type) => {
    switch (type) {
      case 'course': return <Video className="w-4 h-4" />;
      case 'book': return <Book className="w-4 h-4" />;
      case 'tutorial': return <Monitor className="w-4 h-4" />;
      case 'paper': return <FileText className="w-4 h-4" />;
      default: return <BookOpen className="w-4 h-4" />;
    }
  };

  const ResourceCard = ({ resource, onToggle, isCompleted }) => (
    <div className={`border rounded-lg p-4 transition-all duration-200 ${
      isCompleted ? 'bg-green-50 border-green-200' : 'bg-white border-gray-200 hover:shadow-md'
    }`}>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-start gap-2 flex-1">
          <button
            onClick={() => onToggle(resource.id)}
            className={`mt-1 ${isCompleted ? 'text-green-600' : 'text-gray-400'}`}
          >
            {isCompleted ? <CheckCircle2 className="w-4 h-4" /> : <Circle className="w-4 h-4" />}
          </button>
          <div className="flex-1">
            <h4 className={`font-medium ${isCompleted ? 'text-green-900 line-through' : 'text-gray-900'}`}>
              {resource.title}
            </h4>
          </div>
        </div>
        <ExternalLink className="w-4 h-4 text-gray-400 ml-2 flex-shrink-0" />
      </div>
      
      <div className="flex items-center gap-2 mb-2 ml-6">
        <span className="text-sm text-gray-600">{resource.provider}</span>
        <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${
          resource.type === 'course' ? 'bg-blue-100 text-blue-700' :
          resource.type === 'book' ? 'bg-green-100 text-green-700' :
          resource.type === 'tutorial' ? 'bg-purple-100 text-purple-700' :
          'bg-orange-100 text-orange-700'
        }`}>
          {getResourceIcon(resource.type)}
          {resource.type}
        </span>
      </div>
      
      <div className="flex items-center justify-between text-sm ml-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <Star className="w-4 h-4 text-yellow-400 fill-current" />
            <span>{resource.rating}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="w-4 h-4 text-gray-400" />
            <span>{resource.duration}</span>
          </div>
        </div>
        <span className="text-gray-600 font-medium">{resource.cost}</span>
      </div>
    </div>
  );

  const SkillSection = ({ skill, onToggle, isCompleted }) => (
    <div className={`border rounded-lg transition-all duration-200 ${
      isCompleted ? 'bg-green-50 border-green-200' : 'bg-white border-gray-200'
    }`}>
      <div 
        className="p-4 cursor-pointer"
        onClick={() => onToggle(skill.id)}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            {isCompleted ? (
              <CheckCircle2 className="w-5 h-5 text-green-600" />
            ) : (
              <Circle className="w-5 h-5 text-gray-400" />
            )}
            <div>
              <h3 className={`font-semibold ${isCompleted ? 'text-green-900 line-through' : 'text-gray-900'}`}>
                {skill.title}
              </h3>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <span className="capitalize">{skill.category}</span>
                <span>•</span>
                <span>{skill.timeRequired}</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  skill.importance === 'Critical' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
                }`}>
                  {skill.importance}
                </span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="ml-8">
          <div className="mb-3">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Topics to Learn:</h4>
            <div className="flex flex-wrap gap-2">
              {skill.topics.map((topic, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                >
                  {topic}
                </span>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-3">Learning Resources:</h4>
            <div className="space-y-3">
              {skill.resources.map((resource) => (
                <ResourceCard
                  key={resource.id}
                  resource={resource}
                  onToggle={toggleResource}
                  isCompleted={completedResources.has(resource.id)}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const selectedPhaseData = roadmapData.phases[selectedPhase];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <Brain className="text-blue-600" />
                AI Engineer Transition Roadmap
              </h1>
              <p className="text-gray-600 mt-2">
                {roadmapData.metadata.duration} • {roadmapData.metadata.timeCommitment} • 
                <span className="ml-2 text-blue-600 font-medium">v{roadmapData.metadata.version}</span>
              </p>
            </div>
            
            {/* Study Timer */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-center">
                <div className="text-2xl font-mono font-bold text-gray-900 mb-2">
                  {formatTime(studyTime)}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsTimerActive(!isTimerActive)}
                    className={`px-3 py-1 rounded-md text-sm font-medium flex items-center gap-1 ${
                      isTimerActive 
                        ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                        : 'bg-green-100 text-green-700 hover:bg-green-200'
                    }`}
                  >
                    {isTimerActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    {isTimerActive ? 'Pause' : 'Start'}
                  </button>
                  <button
                    onClick={() => { setStudyTime(0); setIsTimerActive(false); }}
                    className="px-3 py-1 rounded-md text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 flex items-center gap-1"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Reset
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Phase Selection Sidebar */}
          <div className="lg:col-span-1">
            <div className="sticky top-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Target className="text-blue-600" />
                Learning Phases
              </h2>
              
              <div className="space-y-3 mb-6">
                {roadmapData.phases.map((phase, index) => (
                  <PhaseCard
                    key={phase.id}
                    phase={phase}
                    isSelected={selectedPhase === index}
                    onClick={() => setSelectedPhase(index)}
                  />
                ))}
              </div>

              {/* Overall Progress */}
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <h3 className="font-semibold text-gray-900 mb-3">Overall Progress</h3>
                <div className="flex items-center justify-center">
                  <ProgressRing 
                    progress={getOverallProgress()} 
                    size={80} 
                  />
                </div>
                <div className="text-center mt-2 text-sm text-gray-600">
                  {completedMilestones.size + completedSkills.size + completedResources.size} items completed
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Phase Header */}
            <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
              <div className="flex items-center gap-4 mb-4">
                <div className={`w-12 h-12 rounded-full ${selectedPhaseData.color} flex items-center justify-center text-white font-bold text-xl`}>
                  {selectedPhaseData.icon}
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{selectedPhaseData.title}</h2>
                  <p className="text-gray-600">{selectedPhaseData.description}</p>
                </div>
              </div>
              
              <div className="grid grid-cols-4 gap-4 text-center">
                <div className="p-3 bg-gray-50 rounded-lg">
                  <Clock className="w-5 h-5 text-gray-600 mx-auto mb-1" />
                  <div className="text-sm font-medium text-gray-900">{selectedPhaseData.duration}</div>
                  <div className="text-xs text-gray-600">Duration</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <Zap className="w-5 h-5 text-gray-600 mx-auto mb-1" />
                  <div className="text-sm font-medium text-gray-900">{selectedPhaseData.totalHours}</div>
                  <div className="text-xs text-gray-600">Total Hours</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-gray-600 mx-auto mb-1" />
                  <div className="text-sm font-medium text-gray-900">{selectedPhaseData.priority}</div>
                  <div className="text-xs text-gray-600">Priority</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <Award className="w-5 h-5 text-gray-600 mx-auto mb-1" />
                  <div className="text-sm font-medium text-gray-900">{getPhaseProgress(selectedPhaseData)}%</div>
                  <div className="text-xs text-gray-600">Complete</div>
                </div>
              </div>
            </div>

            {/* Skills Section */}
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Code className="text-blue-600" />
                Key Skills & Resources
              </h3>
              
              <div className="space-y-4">
                {selectedPhaseData.keySkills.map((skill) => (
                  <SkillSection
                    key={skill.id}
                    skill={skill}
                    onToggle={toggleSkill}
                    isCompleted={completedSkills.has(skill.id)}
                  />
                ))}
              </div>
            </div>

            {/* Milestones Section */}
            <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <CheckCircle2 className="text-purple-600" />
                Phase Milestones
              </h3>
              
              <div className="space-y-3">
                {selectedPhaseData.milestones.map((milestone, index) => {
                  const isCompleted = completedMilestones.has(`${selectedPhaseData.id}-${index}`);
                  
                  return (
                    <div
                      key={index}
                      className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                        isCompleted 
                          ? 'bg-green-50 border-green-200' 
                          : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                      }`}
                      onClick={() => toggleMilestone(selectedPhaseData.id, index)}
                    >
                      {isCompleted ? (
                        <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
                      ) : (
                        <Circle className="w-5 h-5 text-gray-400 flex-shrink-0" />
                      )}
                      
                      <div className="flex-1">
                        <div className={`font-medium ${isCompleted ? 'text-green-900 line-through' : 'text-gray-900'}`}>
                          {milestone.text}
                        </div>
                        <div className="text-sm text-gray-600">Week {milestone.week}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Projects Section */}
            {selectedPhaseData.projects && (
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <Lightbulb className="text-orange-600" />
                  Suggested Projects
                </h3>
                
                <div className="space-y-4">
                  {selectedPhaseData.projects.map((project, index) => (
                    <div key={index} className="border border-gray-100 rounded-lg p-4">
                      <h4 className="font-semibold text-gray-900 mb-2">{project.title}</h4>
                      <p className="text-gray-600 mb-3">{project.description}</p>
                      <div>
                        <h5 className="text-sm font-medium text-gray-700 mb-2">Deliverables:</h5>
                        <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                          {project.deliverables.map((deliverable, idx) => (
                            <li key={idx}>{deliverable}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Career Info Footer */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
            <DollarSign className="w-8 h-8 text-green-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Salary Progression</h3>
            <div className="space-y-1 text-sm">
              <div className="text-green-600 font-medium">Entry: {roadmapData.careerInfo.salaryRanges.entryLevel}</div>
              <div className="text-green-700 font-medium">Mid: {roadmapData.careerInfo.salaryRanges.experienced}</div>
              <div className="text-green-800 font-bold">Senior: {roadmapData.careerInfo.salaryRanges.senior}</div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
            <Users className="w-8 h-8 text-blue-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Job Market</h3>
            <div className="text-lg font-bold text-blue-600">{roadmapData.careerInfo.jobMarket.demand} Demand</div>
            <div className="text-sm text-gray-600">{roadmapData.careerInfo.jobMarket.growth} growth</div>
          </div>
          
          <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
            <Globe className="w-8 h-8 text-purple-600 mx-auto mb-3" />
            <h3 className="font-semibold text-gray-900 mb-2">Remote Opportunities</h3>
            <div className="text-lg font-bold text-purple-600">{roadmapData.careerInfo.jobMarket.remoteWork}</div>
            <div className="text-sm text-gray-600">of AI positions</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AiEngineerRoadmap;