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

#pragma once

#include <stdint.h>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>
#include <vector>

namespace gtl
{
  class GbThreadPool
  {
  public:
    static uint32_t GetRecommendedThreadCount();

  public:
    GbThreadPool(const GbThreadPool&) = delete;
    GbThreadPool& operator=(const GbThreadPool&) = delete;

    GbThreadPool(uint32_t threadCount);

    ~GbThreadPool();

    uint32_t getThreadCount() const;

    template<typename Function>
    void parallelFor(uint32_t count, Function function)
    {
      if (count == 0)
      {
        return;
      }

      uint32_t participantCount = std::min(count, getThreadCount());
      if (participantCount == 0 || count == 1)
      {
        for (uint32_t i = 0; i < count; i++)
        {
          function(i);
        }
        return;
      }

      struct Job
      {
        Job(Function function_, uint32_t count_)
          : function(std::move(function_))
          , count(count_)
        {
        }

        Function function;
        uint32_t count;
        std::atomic<uint32_t> nextIndex = 0;
        std::atomic<uint32_t> pendingTaskCount = 0;
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
      };

      auto job = std::make_shared<Job>(std::move(function), count);
      job->pendingTaskCount = participantCount;

      for (uint32_t i = 0; i < participantCount; i++)
      {
        enqueue([job]() {
          uint32_t index;
          while ((index = job->nextIndex.fetch_add(1, std::memory_order_relaxed)) < job->count)
          {
            job->function(index);
          }

          if (job->pendingTaskCount.fetch_sub(1, std::memory_order_acq_rel) == 1)
          {
            std::lock_guard<std::mutex> lock(job->mutex);
            job->done = true;
            job->cv.notify_all();
          }
        });
      }

      std::unique_lock<std::mutex> lock(job->mutex);
      job->cv.wait(lock, [&job]() { return job->done; });
    }

  private:
    void enqueue(std::function<void()> task);
    void workerLoop();

  private:
    std::vector<std::thread> m_threads;
    std::deque<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stopping = false;
  };
}
