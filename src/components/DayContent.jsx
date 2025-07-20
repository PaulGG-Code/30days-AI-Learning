import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Menu, ChevronLeft, ChevronRight, BookOpen, Code, Play, CheckCircle, Clock, Brain, Lightbulb, Target, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import ReactMarkdown from 'react-markdown'
import { useNavigate } from 'react-router-dom'
import PracticeTab from './PracticeTab.jsx'
import remarkGfm from 'remark-gfm'

const typeColors = {
  foundation: 'bg-blue-100 text-blue-800 border-blue-200',
  'deep-learning': 'bg-purple-100 text-purple-800 border-purple-200',
  llm: 'bg-green-100 text-green-800 border-green-200',
  agents: 'bg-orange-100 text-orange-800 border-orange-200',
  ethics: 'bg-red-100 text-red-800 border-red-200',
  future: 'bg-indigo-100 text-indigo-800 border-indigo-200',
  'hands-on': 'bg-yellow-100 text-yellow-800 border-yellow-200',
  philosophy: 'bg-pink-100 text-pink-800 border-pink-200',
  reflection: 'bg-gray-100 text-gray-800 border-gray-200',
  challenge: 'bg-emerald-100 text-emerald-800 border-emerald-200'
}

const typeIcons = {
  foundation: Brain,
  'deep-learning': Zap,
  llm: BookOpen,
  agents: Target,
  ethics: Lightbulb,
  future: Lightbulb,
  'hands-on': Code,
  philosophy: Brain,
  reflection: Lightbulb,
  challenge: Target
}

// Dynamically import all markdown files for days 1-30
const dayMdFiles = import.meta.glob('/src/book/day-*/README.md', { as: 'raw' })
// Dynamically import all example code files for days 1-30 (py, js, jsx, md, txt)
const exampleFiles = import.meta.glob('/src/examples/day-*/**/*.{py,js,jsx,md,txt}', { as: 'raw' })

export default function DayContent({ day, courseData, setSidebarOpen, setCurrentDay, activeTab, setActiveTab }) {
  const [isCompleted, setIsCompleted] = useState(false)
  const [content, setContent] = useState('Loading...')
  const [examples, setExamples] = useState([])
  const navigate = useNavigate()

  const lesson = courseData.dailyLessons[day]
  const TypeIcon = typeIcons[lesson?.type] || BookOpen

  useEffect(() => {
    const dayStr = String(day).padStart(2, '0')
    const fileKey = `/src/book/day-${dayStr}/README.md`
    // Debug logs
    console.log('Book keys:', Object.keys(dayMdFiles))
    console.log('Example keys:', Object.keys(exampleFiles))
    console.log('Looking for book fileKey:', fileKey)
    if (dayMdFiles[fileKey]) {
      dayMdFiles[fileKey]().then(setContent)
    } else {
      setContent('Content coming soon...')
    }
    // Load example files for this day
    const prefix = `/src/examples/day-${dayStr}/`
    const files = Object.keys(exampleFiles).filter(key => key.startsWith(prefix))
    console.log('Example file prefix:', prefix, 'Found:', files)
    if (files.length > 0) {
      Promise.all(files.map(f => exampleFiles[f]().then(content => ({ name: f.split('/').pop(), content })))).then(setExamples)
    } else {
      setExamples([])
    }
  }, [day])

  const handleComplete = () => {
    setIsCompleted(!isCompleted)
    // In a real app, this would update the course progress
  }

  const goToPreviousDay = () => {
    if (day > 1) {
      if (setCurrentDay) setCurrentDay(day - 1)
      navigate(`/course/day/${day - 1}`)
    }
  }

  const goToNextDay = () => {
    if (day < courseData.totalDays) {
      if (setCurrentDay) setCurrentDay(day + 1)
      navigate(`/course/day/${day + 1}`)
    }
  }

  return (
    <div className="flex-1 overflow-auto flex flex-col h-full">
      <div className="max-w-4xl mx-auto p-6 flex-1">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="content">
              <BookOpen className="w-4 h-4 mr-2" />
              Content
            </TabsTrigger>
            <TabsTrigger value="examples">
              <Code className="w-4 h-4 mr-2" />
              Examples
            </TabsTrigger>
            <TabsTrigger value="practice">
              <Target className="w-4 h-4 mr-2" />
              Practice
            </TabsTrigger>
            <TabsTrigger value="resources">
              <Lightbulb className="w-4 h-4 mr-2" />
              Resources
            </TabsTrigger>
          </TabsList>

          <TabsContent value="content" className="mt-6">
            <div className="mb-4">
              <Button variant="outline" onClick={() => navigate('/')} className="flex items-center gap-2">
                <ChevronLeft className="w-4 h-4" />
                Back to Home
              </Button>
            </div>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TypeIcon className={`w-6 h-6 ${typeColors[lesson?.type] || ''}`} />
                  {lesson?.title || `Day ${day}`}
                </CardTitle>
                <CardDescription>{lesson?.type}</CardDescription>
              </CardHeader>
              <CardContent className="p-8">
                <div className="prose prose-lg max-w-3xl mx-auto bg-white/95 rounded-xl shadow-lg p-8 my-8 border border-gray-300 text-gray-900" style={{ fontSize: '1.15rem', lineHeight: '1.8' }}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      pre: ({node, ...props}) => (
                        <pre {...props} className="overflow-x-auto bg-gray-900 text-green-400 p-4 rounded text-sm" />
                      ),
                      code: ({node, ...props}) => (
                        <code {...props} style={{ wordBreak: 'break-word', whiteSpace: 'pre' }} />
                      ),
                      table: ({node, ...props}) => (
                        <div className="w-full overflow-x-auto">
                          <table {...props} className="min-w-[600px] border border-gray-300 border-collapse" />
                        </div>
                      ),
                      th: ({node, ...props}) => (
                        <th {...props} className="border border-gray-300 px-2 py-1 bg-gray-100" />
                      ),
                      td: ({node, ...props}) => (
                        <td {...props} className="border border-gray-200 px-2 py-1" />
                      ),
                    }}
                  >
                    {content}
                  </ReactMarkdown>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="examples" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Example</CardTitle>
              </CardHeader>
              <CardContent>
                {examples.length === 0 ? (
                  <p>No example code for this day.</p>
                ) : (
                  examples.map((ex, i) => (
                    <div key={i} className="mb-6">
                      <div className="font-mono text-xs text-gray-500 mb-2">{ex.name}</div>
                      <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto">
                        <code>{ex.content}</code>
                      </pre>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="practice" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Practice Exercise</CardTitle>
              </CardHeader>
              <CardContent>
                {/* Find the first Python example for the day, if any */}
                {examples.find(ex => ex.name && ex.name.endsWith('.py')) ? (
                  <PracticeTab initialCode={examples.find(ex => ex.name && ex.name.endsWith('.py')).content} />
                ) : (
                  <p>Practical exercise coming soon...</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="resources" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Additional Reading</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    <li>
                      <a href="#" className="text-blue-600 hover:underline">
                        "What is Artificial Intelligence?" - Stanford AI
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-blue-600 hover:underline">
                        "A Brief History of AI" - MIT Technology Review
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-blue-600 hover:underline">
                        "AI Applications in Daily Life" - Harvard Business Review
                      </a>
                    </li>
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Next Steps</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600 mb-4">Next steps coming soon...</p>
                  <Button variant="outline" className="w-full">
                    Preview Tomorrow's Lesson
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
      {/* Navigation Footer */}
      <footer className="bg-white border-t border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={goToPreviousDay}
            disabled={day === 1}
            className="flex items-center"
          >
            <ChevronLeft className="w-4 h-4 mr-2" />
            Previous Day
          </Button>

          <div className="text-sm text-gray-500">
            Day {day} of {courseData.totalDays}
          </div>

          <Button
            onClick={goToNextDay}
            disabled={day === courseData.totalDays}
            className="flex items-center"
          >
            Next Day
            <ChevronRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </footer>
    </div>
  )
}

