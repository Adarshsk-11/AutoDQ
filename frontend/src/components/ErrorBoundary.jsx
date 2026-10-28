import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    // In a production build this is where you'd send the error to a
    // logging service. For the MVP, surfacing it to the console is enough.
    console.error('AutoDQ UI crashed:', error, info)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
    this.props.onReset?.()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-lg mx-auto mt-16 bg-white border border-red-200 rounded-2xl p-8 text-center">
          <h2 className="text-lg font-bold text-red-700 mb-2">Something went wrong displaying this page</h2>
          <p className="text-sm text-slate-500 mb-4">
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          <button
            onClick={this.handleReset}
            className="px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium"
          >
            Start over
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
