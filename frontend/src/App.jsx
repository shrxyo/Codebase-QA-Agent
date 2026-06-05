import { useState } from 'react'
import IndexPanel from './components/IndexPanel.jsx'
import RepoSummaryCard from './components/RepoSummaryCard.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import StatusBar from './components/StatusBar.jsx'

export default function App() {
  const [indexedRepo, setIndexedRepo] = useState(null)
  const [indexStats, setIndexStats] = useState(null)
  const [repoSummary, setRepoSummary] = useState(null)
  const [showSummary, setShowSummary] = useState(false)
  const [initialQuestion, setInitialQuestion] = useState(null)

  function handleIndexSuccess({ repo, stats, summary }) {
    setIndexedRepo(repo)
    setIndexStats(stats)
    setRepoSummary(summary)
    setShowSummary(true)
    setInitialQuestion(null)
  }

  function handleContinue(question) {
    setInitialQuestion(question ?? null)
    setShowSummary(false)
  }

  function renderMain() {
    if (!indexedRepo) {
      return <IndexPanel onSuccess={handleIndexSuccess} />
    }

    if (showSummary) {
      return (
        <RepoSummaryCard
          summary={repoSummary}
          stats={indexStats}
          onContinue={handleContinue}
        />
      )
    }

    return <ChatPanel repo={indexedRepo} summary={repoSummary} stats={indexStats} initialQuestion={initialQuestion} />
  }

  return (
    <div style={styles.root}>
      <div style={styles.main}>{renderMain()}</div>
      <StatusBar repo={indexedRepo} stats={indexStats} />
    </div>
  )
}

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    width: '100%',
    overflow: 'hidden',
  },
  main: {
    flex: 1,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
}
