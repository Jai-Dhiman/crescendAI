import { createFileRoute } from '@tanstack/react-router'
import AppChat from '../components/AppChat'

export const Route = createFileRoute('/app/c/$conversationId')({
  component: () => {
    const { conversationId } = Route.useParams()
    return <AppChat initialConversationId={conversationId} />
  },
})
