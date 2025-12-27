import logging

from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class ConversationalAssistant:
    def __init__(self, client: OpenAI):
        self.client = client

    def send_message(self):
        logger.info("Sending message with prebuilt prompt...")

        try:
            response = self.client.responses.create(
                prompt={
                    "id": "pmpt_693207f1c4608194835d7d830d4122c70f68a4444ac8b3cf",
                    "version": "3",
                }
            )
            logger.debug("Raw response: %s", response)
            return response
        except Exception:
            logger.exception("Failed to send message")
            raise


if __name__ == "__main__":
    from logging_setup import setup_logging

    setup_logging()

    client = OpenAI(api_key=settings.openai_api_key)
    assistant = ConversationalAssistant(client=client)
    assistant.send_message()
