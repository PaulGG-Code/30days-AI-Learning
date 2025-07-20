import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate, useParams, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { BookOpen, Brain, Cpu, Users, Lightbulb, Code, Rocket, Heart, MessageCircle, Star, Users as UsersIcon, Terminal, Globe, User } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import Sidebar from './components/Sidebar'
import DayContent from './components/DayContent'
import './App.css'

// Course data structure
const courseData = {
  title: "AI, LLM & Agent Course",
  subtitle: "A 30-Day Journey from Basics to Building Your Own Models",
  description: "Master the fundamentals of Artificial Intelligence, Large Language Models, and AI Agents through an immersive storytelling approach.",
  totalDays: 30,
  weeks: [
    {
      id: 1,
      title: "Foundations of AI",
      description: "Understanding the basics of artificial intelligence and machine learning",
      days: [1, 2, 3, 4, 5, 6, 7],
      icon: Brain,
      color: "bg-blue-500"
    },
    {
      id: 2,
      title: "Deep Learning & Neural Networks",
      description: "Diving into neural networks and deep learning architectures",
      days: [8, 9, 10, 11, 12, 13, 14],
      icon: Cpu,
      color: "bg-purple-500"
    },
    {
      id: 3,
      title: "Large Language Models",
      description: "Exploring the giants of language and their capabilities",
      days: [15, 16, 17, 18, 19, 20, 21],
      icon: BookOpen,
      color: "bg-green-500"
    },
    {
      id: 4,
      title: "AI Agents & Ethics",
      description: "Building autonomous agents and responsible AI practices",
      days: [22, 23, 24, 25, 26, 27, 28],
      icon: Users,
      color: "bg-orange-500"
    },
    {
      id: 5,
      title: "Future & Mastery",
      description: "Looking ahead and consolidating your knowledge",
      days: [29, 30],
      icon: Rocket,
      color: "bg-red-500"
    }
  ],
  dailyLessons: {
    1: { title: "What is AI?", type: "foundation", completed: false },
    2: { title: "The Building Blocks: Data and Algorithms", type: "foundation", completed: false },
    3: { title: "Supervised Learning: Learning from Examples", type: "foundation", completed: false },
    4: { title: "Unsupervised Learning: Discovering Hidden Patterns", type: "foundation", completed: false },
    5: { title: "Evaluation and Metrics: Knowing if We're Right", type: "foundation", completed: false },
    6: { title: "Introduction to Neural Networks: The Brain-Inspired Machines", type: "foundation", completed: false },
    7: { title: "Weekend Challenge & Reflection", type: "challenge", completed: false },
    8: { title: "Deep Neural Networks: Going Deeper", type: "deep-learning", completed: false },
    9: { title: "Recurrent Neural Networks (RNNs): Learning Sequences", type: "deep-learning", completed: false },
    10: { title: "Introduction to Natural Language Processing (NLP)", type: "deep-learning", completed: false },
    11: { title: "Word Embeddings: Giving Words Meaning", type: "deep-learning", completed: false },
    12: { title: "Sequence-to-Sequence Models: Translation and Generation", type: "deep-learning", completed: false },
    13: { title: "Transformers: The Revolution in NLP", type: "deep-learning", completed: false },
    14: { title: "Weekend Challenge & Reflection", type: "challenge", completed: false },
    15: { title: "What are LLMs? The Giants of Language", type: "llm", completed: false },
    16: { title: "LLM Architectures: GPT, BERT, and Beyond", type: "llm", completed: false },
    17: { title: "Prompt Engineering: Speaking to the Giants", type: "llm", completed: false },
    18: { title: "Fine-tuning LLMs: Customizing the Knowledge", type: "llm", completed: false },
    19: { title: "Applications of LLMs: Beyond Chatbots", type: "llm", completed: false },
    20: { title: "AI Agents: From LLMs to Autonomous Action", type: "agents", completed: false },
    21: { title: "Designing Agent Workflows", type: "agents", completed: false },
    22: { title: "Multi-Agent Systems: Societies of AI", type: "agents", completed: false },
    23: { title: "Ethical Considerations and Responsible AI", type: "ethics", completed: false },
    24: { title: "Future Trends and Challenges in AI", type: "future", completed: false },
    25: { title: "Building Your Own Simple AI Model", type: "hands-on", completed: false },
    26: { title: "Training Your Own LLM (Conceptual)", type: "hands-on", completed: false },
    27: { title: "Deploying and Managing AI Models", type: "hands-on", completed: false },
    28: { title: "AI and Humanity: The Symbiotic Future", type: "philosophy", completed: false },
    29: { title: "Final Reflections: Key Takeaways", type: "reflection", completed: false },
    30: { title: "Your AI Journey: Next Steps and Resources", type: "reflection", completed: false }
  }
}

const featuredExample = `def simple_ai_program(input_data):
    if "hot" in input_data.lower():
        return "It's a good day for ice cream!"
    elif "cold" in input_data.lower():
        return "Perhaps a hot cup of tea is in order."
    else:
        return "I'm not sure what to recommend."

# Try it out!
print(simple_ai_program("It's hot and sunny today"))`

function HomePage() {
  const [progress, setProgress] = useState(0)
  const [completedDays, setCompletedDays] = useState(0)
  const navigate = useNavigate()

  useEffect(() => {
    // Simulate progress loading
    const timer = setTimeout(() => setProgress(completedDays / courseData.totalDays * 100), 500)
    return () => clearTimeout(timer)
  }, [completedDays])

  // Sample lessons for curriculum timeline
  const weekSamples = courseData.weeks.map(week => ({
    ...week,
    sampleLessons: week.days.slice(0, 3).map(day => courseData.dailyLessons[day]?.title)
  }))

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex flex-col">
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative overflow-hidden bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 text-white"
      >
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 flex flex-col items-center">
          <h1 className="text-4xl sm:text-5xl font-bold mb-4 text-center drop-shadow-lg">AI, LLM & Agent Course</h1>
          <h2 className="text-2xl sm:text-3xl font-semibold mb-4 text-center drop-shadow">A 30-Day Journey from Basics to Building Your Own Models</h2>
          <p className="text-lg sm:text-xl max-w-2xl text-center mb-8 drop-shadow">
            Master the fundamentals of Artificial Intelligence, Large Language Models, and AI Agents through an immersive, hands-on, storytelling approach. No prior experience required!
          </p>
          <div className="flex flex-col sm:flex-row gap-4 mt-4 mb-8">
            <Button size="lg" className="bg-white text-blue-600 hover:bg-blue-50 px-8 py-3 text-lg font-semibold shadow" onClick={() => navigate('/course/day/1')}>
              <BookOpen className="w-5 h-5 mr-2" />
              Start Learning
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="border-2 border-white text-white font-bold hover:bg-white/10 px-8 py-3 text-lg shadow"
              onClick={() => navigate('/course/day/1?tab=examples')}
            >
              <Code className="w-5 h-5 mr-2" />
              View Examples
            </Button>
          </div>
          {/* Progress Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="mt-6 max-w-md mx-auto"
          >
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Course Progress</span>
                <span className="text-sm">{completedDays}/{courseData.totalDays} days</span>
              </div>
              <Progress value={progress} className="h-2 bg-white/20" />
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* How it Works Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h3 className="text-2xl font-bold mb-4 text-center">How It Works</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div className="flex flex-col items-center">
            <Brain className="w-8 h-8 text-blue-600 mb-2" />
            <span className="font-semibold">Learn AI Concepts</span>
            <span className="text-gray-600 text-sm">Understand the foundations of AI, ML, and LLMs with clear, story-driven lessons.</span>
          </div>
          <div className="flex flex-col items-center">
            <Code className="w-8 h-8 text-purple-600 mb-2" />
            <span className="font-semibold">Hands-On Practice</span>
            <span className="text-gray-600 text-sm">Run and edit real code examples, and test your knowledge with interactive exercises.</span>
          </div>
          <div className="flex flex-col items-center">
            <Rocket className="w-8 h-8 text-green-600 mb-2" />
            <span className="font-semibold">Build Projects</span>
            <span className="text-gray-600 text-sm">Apply your skills to build and deploy your own AI models and agents.</span>
          </div>
          <div className="flex flex-col items-center">
            <Heart className="w-8 h-8 text-pink-600 mb-2" />
            <span className="font-semibold">Join a Community</span>
            <span className="text-gray-600 text-sm">Share progress, ask questions, and learn together with fellow explorers.</span>
          </div>
        </div>
      </div>

      {/* Course Highlights Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h3 className="text-2xl font-bold mb-4 text-center">What You'll Learn</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 text-center">
          <div className="flex flex-col items-center">
            <Star className="w-8 h-8 text-yellow-500 mb-2" />
            <span className="font-semibold">AI & ML Foundations</span>
            <span className="text-gray-600 text-sm">Understand the core concepts of artificial intelligence and machine learning.</span>
          </div>
          <div className="flex flex-col items-center">
            <Terminal className="w-8 h-8 text-blue-500 mb-2" />
            <span className="font-semibold">Hands-On Coding</span>
            <span className="text-gray-600 text-sm">Write, run, and experiment with real Python code and AI models.</span>
          </div>
          <div className="flex flex-col items-center">
            <BookOpen className="w-8 h-8 text-green-500 mb-2" />
            <span className="font-semibold">Large Language Models</span>
            <span className="text-gray-600 text-sm">Explore LLMs, prompt engineering, and fine-tuning techniques.</span>
          </div>
          <div className="flex flex-col items-center">
            <UsersIcon className="w-8 h-8 text-orange-500 mb-2" />
            <span className="font-semibold">Build AI Agents</span>
            <span className="text-gray-600 text-sm">Create autonomous agents and multi-agent systems.</span>
          </div>
          <div className="flex flex-col items-center">
            <Lightbulb className="w-8 h-8 text-indigo-500 mb-2" />
            <span className="font-semibold">Ethics & Responsible AI</span>
            <span className="text-gray-600 text-sm">Learn about AI safety, bias, and ethical considerations.</span>
          </div>
          <div className="flex flex-col items-center">
            <Globe className="w-8 h-8 text-pink-500 mb-2" />
            <span className="font-semibold">Deploy & Share</span>
            <span className="text-gray-600 text-sm">Deploy your models and share your work with the world.</span>
          </div>
        </div>
      </div>

      {/* Curriculum Timeline Section */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h3 className="text-2xl font-bold mb-4 text-center">Curriculum Timeline</h3>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
          {weekSamples.map(week => (
            <div key={week.id} className="bg-white rounded-lg shadow p-4 flex flex-col items-center">
              <div className={`inline-flex items-center justify-center w-12 h-12 ${week.color} rounded-lg mb-2`}>
                <week.icon className="w-6 h-6 text-white" />
              </div>
              <div className="font-bold mb-1 text-center">Week {week.id}</div>
              <div className="text-sm text-gray-700 mb-2 text-center">{week.title}</div>
              <ul className="text-xs text-gray-500 list-disc list-inside text-left">
                {week.sampleLessons.map((title, i) => (
                  <li key={i}>{title}</li>
                ))}
                {week.days.length > 3 && <li>...and more</li>}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Student Testimonials Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h3 className="text-2xl font-bold mb-4 text-center">What Students Say</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow p-6 flex flex-col items-center">
            <User className="w-10 h-10 text-blue-500 mb-2" />
            <div className="italic text-gray-700 mb-2">“This course made AI approachable and fun. The hands-on code and stories kept me engaged every day!”</div>
            <div className="font-semibold text-sm text-gray-600">— Alex, Student</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 flex flex-col items-center">
            <User className="w-10 h-10 text-green-500 mb-2" />
            <div className="italic text-gray-700 mb-2">“I built my first AI agent and deployed it! The interactive practice and real-world projects are amazing.”</div>
            <div className="font-semibold text-sm text-gray-600">— Jamie, Developer</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 flex flex-col items-center">
            <User className="w-10 h-10 text-purple-500 mb-2" />
            <div className="italic text-gray-700 mb-2">“The best part is the community. I always got help and feedback on my projects.”</div>
            <div className="font-semibold text-sm text-gray-600">— Taylor, Data Scientist</div>
          </div>
        </div>
      </div>

      {/* Featured Projects Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h3 className="text-2xl font-bold mb-4 text-center">Featured Projects</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow p-6 flex flex-col items-center">
            <Terminal className="w-10 h-10 text-blue-600 mb-2" />
            <div className="font-bold mb-1">Text Classifier</div>
            <div className="text-gray-600 text-sm text-center">Train a model to classify news articles or movie reviews using real data.</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 flex flex-col items-center">
            <Rocket className="w-10 h-10 text-green-600 mb-2" />
            <div className="font-bold mb-1">AI Agent Demo</div>
            <div className="text-gray-600 text-sm text-center">Build and deploy an autonomous agent that can search, plan, and act.</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 flex flex-col items-center">
            <BookOpen className="w-10 h-10 text-purple-600 mb-2" />
            <div className="font-bold mb-1">LLM Playground</div>
            <div className="text-gray-600 text-sm text-center">Experiment with prompt engineering and fine-tuning on large language models.</div>
          </div>
        </div>
      </div>

      {/* FAQ Section */}
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h3 className="text-2xl font-bold mb-4 text-center">Frequently Asked Questions</h3>
        <div className="space-y-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="font-semibold mb-1">Do I need prior experience?</div>
            <div className="text-gray-600 text-sm">No! The course is beginner-friendly and guides you step by step.</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="font-semibold mb-1">How much time does it take?</div>
            <div className="text-gray-600 text-sm">Each day’s lesson is designed to take 30–60 minutes, but you can go at your own pace.</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="font-semibold mb-1">Is it free?</div>
            <div className="text-gray-600 text-sm">Yes! All course materials and code examples are open and free to use.</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="font-semibold mb-1">What tools do I need?</div>
            <div className="text-gray-600 text-sm">Just a web browser. All code runs in the browser or in Google Colab—no setup required.</div>
          </div>
        </div>
      </div>

      {/* Meet the Instructor Section */}
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-12 flex flex-col items-center">
        <img src="https://avatars.githubusercontent.com/u/9919?v=4" alt="Instructor" className="w-24 h-24 rounded-full mb-4 border-4 border-blue-200 shadow" />
        <div className="font-bold text-lg mb-1">Paul Gedeon</div>
        <div className="text-gray-600 text-center mb-2">AI Educator & Developer</div>
        <div className="text-gray-700 text-center mb-2">Passionate about making AI accessible to everyone. I love building tools, teaching, and helping others learn and create with AI.</div>
      </div>

      {/* Community Call to Action Section */}
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-12 flex flex-col items-center">
        <h3 className="text-2xl font-bold mb-2 text-center">Join Our Community</h3>
        <p className="text-gray-700 text-center mb-4">Get support, share your progress, and connect with fellow learners in our Discord community.</p>
        <a href="#" className="px-6 py-3 bg-blue-600 text-white rounded font-semibold shadow hover:bg-blue-700 transition">Join Discord</a>
      </div>

      {/* Featured Example Section */}
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h3 className="text-xl font-bold mb-2">Featured Example: Day 1</h3>
        <div className="bg-gray-900 text-green-400 rounded p-4 font-mono text-sm overflow-x-auto">
          <pre style={{ whiteSpace: 'pre-wrap' }}>{featuredExample}</pre>
        </div>
      </div>
    </div>
  )
}

function CoursePageWrapper() {
  // This wrapper redirects /course to /course/day/1
  return <Navigate to="/course/day/1" replace />
}

function CourseDayPage() {
  const params = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [currentDay, setCurrentDay] = useState(Number(params.dayId) || 1)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeTab, setActiveTab] = useState(location.search.includes('tab=examples') ? 'examples' : 'lessons')

  // Sync state with URL param
  useEffect(() => {
    const dayNum = Number(params.dayId)
    if (!isNaN(dayNum) && dayNum !== currentDay) {
      setCurrentDay(dayNum)
    }
    // eslint-disable-next-line
  }, [params.dayId])

  // When currentDay changes, update the URL
  useEffect(() => {
    if (Number(params.dayId) !== currentDay) {
      navigate(`/course/day/${currentDay}`, { replace: true })
    }
    // eslint-disable-next-line
  }, [currentDay])

  // Update active tab when URL changes
  useEffect(() => {
    const urlTab = new URLSearchParams(location.search).get('tab')
    if (urlTab && urlTab !== activeTab) {
      setActiveTab(urlTab)
    }
  }, [location.search, activeTab])

  return (
    <div className="flex h-screen">
      <Sidebar 
        courseData={courseData}
        currentDay={currentDay}
        setCurrentDay={setCurrentDay}
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
      />
      <main className="flex-1 overflow-auto">
        <DayContent 
          day={currentDay}
          courseData={courseData}
          setSidebarOpen={setSidebarOpen}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />
      </main>
    </div>
  )
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/course" element={<CoursePageWrapper />} />
          <Route path="/course/day/:dayId" element={<CourseDayPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App

