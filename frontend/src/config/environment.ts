/// <reference types="vite/client" />

export const ENV_CONFIG = {
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
}
