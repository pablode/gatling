//
// Copyright (C) 2025 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "ThreadPool.h"

namespace gtl
{
  uint32_t GbThreadPool::GetRecommendedThreadCount()
  {
    uint32_t threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0)
    {
      threadCount = 4;
    }
    return threadCount;
  }

  GbThreadPool::GbThreadPool(uint32_t threadCount)
  {
    m_threads.reserve(threadCount);

    for (uint32_t i = 0; i < threadCount; i++)
    {
      m_threads.emplace_back([this]() { workerLoop(); });
    }
  }

  GbThreadPool::~GbThreadPool()
  {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_stopping = true;
    }
    m_cv.notify_all();

    for (std::thread& thread : m_threads)
    {
      thread.join();
    }
  }

  uint32_t GbThreadPool::getThreadCount() const
  {
    return (uint32_t) m_threads.size();
  }

  void GbThreadPool::enqueue(std::function<void()> task)
  {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_tasks.push_back(std::move(task));
    }
    m_cv.notify_one();
  }

  void GbThreadPool::workerLoop()
  {
    while (true)
    {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]() { return m_stopping || !m_tasks.empty(); });

        if (m_tasks.empty())
        {
          return;
        }

        task = std::move(m_tasks.front());
        m_tasks.pop_front();
      }

      task();
    }
  }
}
