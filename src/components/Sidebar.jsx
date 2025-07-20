import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronLeft, ChevronRight, Menu, X, BookOpen, CheckCircle, Circle, Brain, Cpu, Users, Rocket } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'

const typeColors = {
  foundation: 'bg-blue-100 text-blue-800',
  'deep-learning': 'bg-purple-100 text-purple-800',
  llm: 'bg-green-100 text-green-800',
  agents: 'bg-orange-100 text-orange-800',
  ethics: 'bg-red-100 text-red-800',
  future: 'bg-indigo-100 text-indigo-800',
  'hands-on': 'bg-yellow-100 text-yellow-800',
  philosophy: 'bg-pink-100 text-pink-800',
  reflection: 'bg-gray-100 text-gray-800',
  challenge: 'bg-emerald-100 text-emerald-800'
}

const weekIcons = {
  1: Brain,
  2: Cpu,
  3: BookOpen,
  4: Users,
  5: Rocket
}

export default function Sidebar({ courseData, currentDay, setCurrentDay, isOpen, setIsOpen }) {
  const [expandedWeek, setExpandedWeek] = useState(Math.ceil(currentDay / 7))
  
  const completedDays = Object.values(courseData.dailyLessons).filter(lesson => lesson.completed).length
  const progressPercentage = (completedDays / courseData.totalDays) * 100

  const toggleWeek = (weekId) => {
    setExpandedWeek(expandedWeek === weekId ? null : weekId)
  }

  const selectDay = (day) => {
    setCurrentDay(day)
    if (window.innerWidth < 768) {
      setIsOpen(false)
    }
  }

  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-40 md:hidden"
            onClick={() => setIsOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{
          x: isOpen ? 0 : -320,
          width: isOpen ? 320 : 0
        }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="fixed md:relative z-50 h-full bg-white border-r border-gray-200 shadow-lg md:shadow-none"
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Course Progress</h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
                className="md:hidden"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-600">
                <span>Progress</span>
                <span>{completedDays}/{courseData.totalDays} days</span>
              </div>
              <Progress value={progressPercentage} className="h-2" />
            </div>
          </div>

          {/* Course Navigation */}
          <ScrollArea className="flex-1">
            <div className="p-4 space-y-2">
              {courseData.weeks.map((week) => {
                const WeekIcon = weekIcons[week.id]
                const isExpanded = expandedWeek === week.id
                const weekDays = week.days
                const completedInWeek = weekDays.filter(day => courseData.dailyLessons[day]?.completed).length

                return (
                  <div key={week.id} className="space-y-1">
                    <Button
                      variant="ghost"
                      onClick={() => toggleWeek(week.id)}
                      className="w-full justify-between p-3 h-auto text-left hover:bg-gray-50"
                    >
                      <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded-lg ${week.color}`}>
                          <WeekIcon className="w-4 h-4 text-white" />
                        </div>
                        <div>
                          <div className="font-medium text-sm">Week {week.id}</div>
                          <div className="text-xs text-gray-500">{week.title}</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="secondary" className="text-xs">
                          {completedInWeek}/{weekDays.length}
                        </Badge>
                        <motion.div
                          animate={{ rotate: isExpanded ? 90 : 0 }}
                          transition={{ duration: 0.2 }}
                        >
                          <ChevronRight className="w-4 h-4" />
                        </motion.div>
                      </div>
                    </Button>

                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="ml-4 space-y-1">
                            {weekDays.map((day) => {
                              const lesson = courseData.dailyLessons[day]
                              const isActive = currentDay === day
                              const isCompleted = lesson?.completed

                              return (
                                <Button
                                  key={day}
                                  variant={isActive ? "default" : "ghost"}
                                  onClick={() => selectDay(day)}
                                  className={`w-full justify-start p-2 h-auto text-left text-sm ${
                                    isActive ? 'bg-blue-600 text-white' : 'hover:bg-gray-50'
                                  }`}
                                >
                                  <div className="flex items-center space-x-3 w-full">
                                    <div className="flex-shrink-0">
                                      {isCompleted ? (
                                        <CheckCircle className="w-4 h-4 text-green-500" />
                                      ) : (
                                        <Circle className="w-4 h-4 text-gray-400" />
                                      )}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <div className="font-medium">Day {day}</div>
                                      <div className={`text-xs truncate ${
                                        isActive ? 'text-blue-100' : 'text-gray-500'
                                      }`}>
                                        {lesson?.title}
                                      </div>
                                    </div>
                                    {lesson?.type && (
                                      <Badge 
                                        variant="secondary" 
                                        className={`text-xs ${typeColors[lesson.type] || 'bg-gray-100 text-gray-800'}`}
                                      >
                                        {lesson.type}
                                      </Badge>
                                    )}
                                  </div>
                                </Button>
                              )
                            })}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )
              })}
            </div>
          </ScrollArea>

          {/* Footer */}
          <div className="p-4 border-t border-gray-200">
            <div className="text-center">
              <p className="text-xs text-gray-500 mb-2">
                AI, LLM & Agent Course
              </p>
              <p className="text-xs text-gray-400">
                30-Day Journey to AI Mastery
              </p>
            </div>
          </div>
        </div>
      </motion.aside>
    </>
  )
}

