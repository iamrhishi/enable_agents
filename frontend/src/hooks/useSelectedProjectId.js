import { useSearchParams } from 'react-router-dom';

/** Returns the global project id from ?project= URL param, or null. */
export function useSelectedProjectId() {
  const [searchParams] = useSearchParams();
  return searchParams.get('project');
}

export default useSelectedProjectId;
