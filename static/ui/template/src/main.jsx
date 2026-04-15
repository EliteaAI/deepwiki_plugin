import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import DeepWikiApp from './DeepWikiApp.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <DeepWikiApp />
  </StrictMode>,
)
