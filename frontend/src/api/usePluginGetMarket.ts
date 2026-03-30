import { useQuery } from 'react-query'
import { getPlugins, type MarketplacePluginListRequest } from './plugin'

export function usePluginListQuery(request: MarketplacePluginListRequest) {
  return useQuery(['plugins', request], () => getPlugins(request), {
    keepPreviousData: true,
  })
}

export const usePluginGetMarket = usePluginListQuery

