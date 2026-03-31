// test/test_all.cpp — Minimal C++11 test suite for cogpilot.jl headers
// Tests core types and pilot (Deep Tree Echo) modules.
// SPDX-License-Identifier: MIT

#include <cog/cog.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <sstream>

// ─────────────────────────────────────────────────────────────────────────────
// Simple test framework
// ─────────────────────────────────────────────────────────────────────────────
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                                        \
    do {                                                                  \
        ++tests_run;                                                      \
        std::cout << "  [" << tests_run << "] " << #name << "... ";       \
        try {                                                             \
            test_##name();                                                \
            ++tests_passed;                                               \
            std::cout << "PASS" << std::endl;                             \
        } catch (const std::exception& e) {                               \
            std::cout << "FAIL: " << e.what() << std::endl;               \
        } catch (...) {                                                   \
            std::cout << "FAIL: unknown exception" << std::endl;          \
        }                                                                 \
    } while (0)

#define ASSERT_TRUE(cond) \
    if (!(cond)) throw std::runtime_error("assertion failed: " #cond)

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::ostringstream ss; \
        ss << "expected " << (a) << " == " << (b); \
        throw std::runtime_error(ss.str()); \
    }

// ─────────────────────────────────────────────────────────────────────────────
// Core module tests
// ─────────────────────────────────────────────────────────────────────────────

void test_handle_creation() {
    cog::Handle h1 = 1;
    cog::Handle h2 = 2;
    ASSERT_TRUE(h1 != h2);
    ASSERT_TRUE(h1 == static_cast<cog::Handle>(1));
    ASSERT_TRUE(cog::UNDEFINED_HANDLE == 0);
}

void test_atomspace_add_node() {
    cog::AtomSpace as;
    auto h = as.add_node(cog::AtomType::CONCEPT_NODE, "hello");
    ASSERT_TRUE(h != cog::UNDEFINED_HANDLE);
    auto* atom = as.get_atom(h);
    ASSERT_TRUE(atom != nullptr);
    ASSERT_EQ(atom->name, std::string("hello"));
    ASSERT_TRUE(atom->type == cog::AtomType::CONCEPT_NODE);
}

void test_atomspace_add_link() {
    cog::AtomSpace as;
    auto h1 = as.add_node(cog::AtomType::CONCEPT_NODE, "A");
    auto h2 = as.add_node(cog::AtomType::CONCEPT_NODE, "B");
    auto link = as.add_link(cog::AtomType::INHERITANCE_LINK, {h1, h2});
    ASSERT_TRUE(link != cog::UNDEFINED_HANDLE);
    auto* latom = as.get_atom(link);
    ASSERT_TRUE(latom != nullptr);
    ASSERT_EQ(latom->outgoing.size(), static_cast<size_t>(2));
}

void test_atomspace_get_by_type() {
    cog::AtomSpace as;
    as.add_node(cog::AtomType::CONCEPT_NODE, "X");
    as.add_node(cog::AtomType::CONCEPT_NODE, "Y");
    auto nodes = as.get_by_type(cog::AtomType::CONCEPT_NODE);
    ASSERT_TRUE(nodes.size() >= 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pilot module tests (A000081 rooted tree sequence)
// ─────────────────────────────────────────────────────────────────────────────

void test_a000081_sequence() {
    cog::pilot::A000081 seq(10);
    // First few values of A000081: 1, 1, 2, 4, 9, 20, 48, 115, 286, 719
    ASSERT_EQ(seq(1), static_cast<uint64_t>(1));
    ASSERT_EQ(seq(2), static_cast<uint64_t>(1));
    ASSERT_EQ(seq(3), static_cast<uint64_t>(2));
    ASSERT_EQ(seq(4), static_cast<uint64_t>(4));
    ASSERT_EQ(seq(5), static_cast<uint64_t>(9));
    ASSERT_EQ(seq(6), static_cast<uint64_t>(20));
}

void test_a000081_extended() {
    cog::pilot::A000081 seq(12);
    ASSERT_EQ(seq(7), static_cast<uint64_t>(48));
    ASSERT_EQ(seq(8), static_cast<uint64_t>(115));
    ASSERT_EQ(seq(9), static_cast<uint64_t>(286));
    ASSERT_EQ(seq(10), static_cast<uint64_t>(719));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "cogpilot.jl C++ test suite" << std::endl;
    std::cout << "=========================" << std::endl;

    std::cout << "\n--- Core Module ---" << std::endl;
    TEST(handle_creation);
    TEST(atomspace_add_node);
    TEST(atomspace_add_link);
    TEST(atomspace_get_by_type);

    std::cout << "\n--- Pilot Module (A000081) ---" << std::endl;
    TEST(a000081_sequence);
    TEST(a000081_extended);

    std::cout << "\n=========================" << std::endl;
    std::cout << tests_passed << "/" << tests_run << " tests passed" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
