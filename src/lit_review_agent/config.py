"""运行时配置：从环境变量或 .env 加载模型、向量库与切分参数。"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置；密钥从环境变量或 .env 文件加载，未声明的环境变量会被忽略。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = Field(default=None, repr=False)
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_directory: Path = Path("data/chroma")
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    max_search_iterations: int = Field(default=3, ge=1, le=6)
    log_level: str = "INFO"

    def require_api_key(self) -> str:
        """返回已配置的 API Key；未配置时抛出可读的中文错误。"""
        if not self.openai_api_key:
            raise ValueError("未配置 OPENAI_API_KEY。请复制 .env.example 为 .env 并填写模型服务商 API Key。")
        return self.openai_api_key
