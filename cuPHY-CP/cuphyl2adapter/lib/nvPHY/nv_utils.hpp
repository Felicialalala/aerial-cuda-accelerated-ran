/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if !defined(NV_UTILS_HPP_INCLUDED_)
#include "nvlog.hpp"
#include <stdexcept>

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 6) // "L2A.MODULE"

// Create our own "preallocated" queue instead of using std::queue, to avoid
// dynamic memory allocation in the real-time code path.
// See https://nvidia.slack.com/archives/CFZB5GYUV/p1676589483373729 for further discussion.

template <class T>
class nv_preallocated_queue
{
    T *arr = nullptr;         // array to store queue elements
    int capacity;   // maximum capacity of the queue
    int frontIdx;      // front points to the front element in the queue (if any)
    int rearIdx;       // rear points to the last element in the queue
    int count;      // current size of the queue

public:
    nv_preallocated_queue(int capacity);   // constructor
    ~nv_preallocated_queue();            // destructor

    // Rule of five reference https://en.cppreference.com/w/cpp/language/rule_of_three
    nv_preallocated_queue(const nv_preallocated_queue& other) // copy constructor
    : nv_preallocated_queue(other.capacity) {
        // printf("nv_preallocated_queue copy constructor called\n");
        std::memcpy(arr, other.arr, sizeof(T) * other.capacity);
        frontIdx = other.frontIdx;
        rearIdx = other.rearIdx;
        count = other.count;
    };

    nv_preallocated_queue(nv_preallocated_queue&& other) noexcept // move constructor
    {
        // printf("nv_preallocated_queue move constructor called\n");
        arr = other.arr;
        other.arr = nullptr;
        capacity = other.capacity;
        frontIdx = other.frontIdx;
        rearIdx = other.rearIdx;
        count = other.count;
    }

    nv_preallocated_queue& operator=(const nv_preallocated_queue& other) // copy assignment
    {
        // printf("nv_preallocated_queue copy assignment called\n");
        return *this = nv_preallocated_queue(other);
    }

    nv_preallocated_queue& operator=(nv_preallocated_queue&& other) noexcept // move assignment
    {
        // printf("nv_preallocated_queue move assignment called\n");
        std::swap(arr, other.arr);
        capacity = other.capacity;
        frontIdx = other.frontIdx;
        rearIdx = other.rearIdx;
        count = other.count;
        return *this;
    }

    void pop();
    void push(T x);
    T& front();
    int size();
    bool isEmpty();
    bool isFull();
};
// Constructor to initialize a queue
template <class T>
nv_preallocated_queue<T>::nv_preallocated_queue(int capacity)
{
    arr = new T[capacity];
    this->capacity = capacity;
    frontIdx = 0;
    rearIdx = -1;
    count = 0;
}

// Destructor to free a queue
template <class T>
nv_preallocated_queue<T>::~nv_preallocated_queue()
{
    // printf("nv_preallocated_queue destructor called\n");
    delete[] arr;
    capacity =  0;
    frontIdx = 0;
    rearIdx = -1;
    count = 0;
}

// Utility function to dequeue the front element
template <class T>
void nv_preallocated_queue<T>::pop()
{
    // check for queue underflow
    if (isEmpty())
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "queue is already empty while popping!!");
    }
    frontIdx = (frontIdx + 1) % capacity;
    count--;
}

// Utility function to add an item to the queue
template <class T>
void nv_preallocated_queue<T>::push(T item)
{
    // check for queue overflow
    if (isFull())
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "queue is already full while pushing!!");
    }
    rearIdx = (rearIdx + 1) % capacity;
    arr[rearIdx] = item;
    count++;
}

// Utility function to return the front element of the queue
template <class T>
T& nv_preallocated_queue<T>::front()
{
    if (isEmpty())
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "queue is already empty while front!!");
    }
    return arr[frontIdx];
}

// Utility function to return the size of the queue
template <class T>
int nv_preallocated_queue<T>::size() {
    return count;
}

// Utility function to check if the queue is empty or not
template <class T>
bool nv_preallocated_queue<T>::isEmpty() {
    return (size() == 0);
}

// Utility function to check if the queue is full or not
template <class T>
bool nv_preallocated_queue<T>::isFull() {
    return (size() == capacity);
}

#undef TAG
#endif

