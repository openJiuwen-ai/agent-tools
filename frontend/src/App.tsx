import { Routes, Route } from 'react-router-dom'
import { Suspense } from 'react'
import LoginPage from '@/pages/LoginPage'
import PluginMarketPage from '@/pages/PluginMarketPage'

function App() {
  return (
    <Suspense fallback={<div className="flex min-h-dvh items-center justify-center">Loading...</div>}>
      <div className="flex h-dvh min-h-0 flex-col overflow-hidden">
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<PluginMarketPage />} />
        </Routes>
      </div>
    </Suspense>
  )
}

export default App
