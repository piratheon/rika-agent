"""Discord interface — adapter + bot client."""
from src.interfaces.discord.adapter import DiscordAdapter
from src.interfaces.discord.bot import DiscordBot

__all__ = ["DiscordAdapter", "DiscordBot"]
