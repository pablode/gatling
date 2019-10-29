#pragma once

#include <vector>
#include <cstdint>

namespace cgpu
{
  class handle_store
  {
  public:
    handle_store();

    ~handle_store();

  public:
    std::uint64_t create();

    bool is_valid(const std::uint64_t& handle) const;

    void free(const std::uint64_t& handle);

    std::uint32_t extract_index(const std::uint64_t& handle) const;

  private:
    std::uint64_t make_handle(
      const std::uint32_t& version,
      const std::uint32_t& index) const;

  private:
    std::uint64_t mMaxIndex;
    std::vector<std::uint32_t> mVersions;
    std::vector<std::uint32_t> mFreeIndices;
  };
}
