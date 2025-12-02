"""
Local LLM client for Relation Radar.

This module provides a mock LLM client interface that will be replaced
with a real local model (like Qwen) in future versions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Message:
    """A chat message with role and content."""
    
    def __init__(self, role: str, content: str):
        self.role = role  # "system", "user", or "assistant"
        self.content = content
    
    def __repr__(self) -> str:
        return f"Message(role='{self.role}', content='{self.content[:50]}...')"


class LocalLLMClient:
    """
    Mock Local LLM client for development and testing.
    
    This client provides a simple mock implementation that generates
    responses based on the provided context. It will be replaced with
    a real local LLM (like Qwen) in future versions.
    """
    
    def __init__(self, model_name: str = "mock-llm"):
        """
        Initialize the local LLM client.
        
        Args:
            model_name: Name of the model to use (mock for now)
        """
        self.model_name = model_name
        self._is_mock = True
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (ignored in mock)
            
        Returns:
            Generated text response
        """
        if self._is_mock:
            return self._mock_generate(prompt)
        
        # Future: Real LLM generation
        raise NotImplementedError("Real LLM not yet implemented")
    
    def chat(self, messages: List[Message], max_tokens: int = 512) -> str:
        """
        Generate a response in a chat conversation.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            
        Returns:
            Assistant's response
        """
        if self._is_mock:
            return self._mock_chat(messages)
        
        # Future: Real LLM chat
        raise NotImplementedError("Real LLM not yet implemented")
    
    def _mock_generate(self, prompt: str) -> str:
        """
        Generate a mock response based on the prompt.
        
        Analyzes the prompt to generate a contextually relevant response.
        """
        # Extract context and question from prompt
        lines = prompt.strip().split('\n')
        
        # Look for context section
        context_lines = []
        question = ""
        in_context = False
        
        for line in lines:
            if '上下文' in line or 'context' in line.lower() or '相关记录' in line:
                in_context = True
                continue
            elif '问题' in line or 'question' in line.lower():
                in_context = False
                question = line.split('：')[-1].split(':')[-1].strip()
            elif in_context and line.strip():
                context_lines.append(line.strip())
        
        # If no structured format, treat the whole prompt as question
        if not question:
            question = prompt[:100]
        
        # Generate response based on context
        if context_lines:
            # Summarize the context
            response = self._summarize_context(context_lines, question)
        else:
            response = f"抱歉，我没有找到关于「{question[:30]}」的相关记录。请确保已经录入了相关的事件信息。"
        
        return response
    
    def _mock_chat(self, messages: List[Message]) -> str:
        """Generate a mock chat response."""
        # Get the last user message
        user_message = ""
        system_context = ""
        
        for msg in messages:
            if msg.role == "user":
                user_message = msg.content
            elif msg.role == "system":
                system_context = msg.content
        
        # Combine system context and user message for response
        if system_context:
            prompt = f"{system_context}\n\n问题：{user_message}"
        else:
            prompt = user_message
        
        return self._mock_generate(prompt)
    
    def _summarize_context(self, context_lines: List[str], question: str) -> str:
        """
        Generate a summary response based on context and question.
        """
        # Detect question type
        is_mood_question = any(kw in question for kw in ['心情', '情绪', '状态', '怎么样', '感觉'])
        is_activity_question = any(kw in question for kw in ['在做什么', '最近', '活动', '干什么'])
        is_preference_question = any(kw in question for kw in ['喜欢', '偏好', '爱好', '习惯'])
        
        # Extract key information from context
        emotions = []
        activities = []
        preferences = []
        
        for line in context_lines:
            # Extract emotions
            if any(kw in line for kw in ['开心', '高兴', '愉快', '快乐']):
                emotions.append('积极')
            elif any(kw in line for kw in ['压力', '焦虑', '担心', '烦恼', '累']):
                emotions.append('有些压力')
            
            # Extract activities
            if any(kw in line for kw in ['健身', '运动', '锻炼']):
                activities.append('健身运动')
            if any(kw in line for kw in ['吃', '聚餐', '饭']):
                activities.append('聚餐')
            if any(kw in line for kw in ['聊天', '交流', '沟通']):
                activities.append('交流沟通')
            
            # Extract preferences
            if any(kw in line for kw in ['喜欢', '爱']):
                # Extract what they like
                for word in ['安静', '麻辣', '川菜', '猫', '健身']:
                    if word in line:
                        preferences.append(word)
        
        # Build response based on question type
        response_parts = []
        
        if is_mood_question and emotions:
            unique_emotions = list(set(emotions))
            if len(unique_emotions) > 1:
                response_parts.append(f"根据记录，情绪状态有些起伏。既有{unique_emotions[0]}的时候，也有{unique_emotions[1]}的时候。")
            else:
                response_parts.append(f"根据记录，整体情绪状态{unique_emotions[0]}。")
        
        if is_activity_question and activities:
            unique_activities = list(set(activities))
            response_parts.append(f"最近主要活动包括：{'、'.join(unique_activities)}。")
        
        if is_preference_question and preferences:
            unique_preferences = list(set(preferences))
            response_parts.append(f"从记录来看，喜欢{'、'.join(unique_preferences)}。")
        
        # Add general summary from context
        if context_lines:
            # Use first context line as additional info
            response_parts.append("\n\n相关记录摘要：\n" + '\n'.join(f"- {line[:80]}" for line in context_lines[:3]))
        
        if response_parts:
            return '\n'.join(response_parts)
        else:
            return "根据现有记录：\n" + '\n'.join(f"- {line[:80]}" for line in context_lines[:5])


# Global instance
_llm_client = None


def get_llm_client() -> LocalLLMClient:
    """
    Get a global singleton instance of the LLM client.
    
    Returns:
        Global LocalLLMClient instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client
