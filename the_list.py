class ListNode:
    def __init__(self,x):
        self.val=x
        self.next=None

class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

def printLinkedList(head):
    '''
    打印链表
    :type head:ListNode
    :return:
    '''
    while head:
        print(head.val,end='->')
        if __name__ == '__main__':
            head=head.next



class AddTwoNumbers:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head=ListNode(None) #头结点
        tail=head           #尾节点
        carry=0
        while l1 and l2:
            carry,num=divmod(l1.val+l2.val+carry,10)
            tail.next=ListNode(num)   #插入尾部
            tail=tail.next            #更新尾部
            l1=l1.next
            l2=l2.next
        while l1:   #如果l1未处理完毕
            carry,num=divmod(l1.val+carry,10)
            tail.next=ListNode(num)
            tail=tail.next
            l1=l1.next
        while l2:
            carry,num=divmod(l2.val+carry,10)
            tail.next=ListNode(num)
            tail=tail.next
            l2=l2.next
        if carry>0:
            tail.next=ListNode(carry)
        return head.next #返回结果不包括尾节点


    def run(self):
        l1=ListNode(2)
        l1.next=ListNode(4)
        l1.next.next=ListNode(3)

        l2=ListNode(5)
        l2.next=ListNode(6)
        l2.next.next=ListNode(4)

        printLinkedList(self.addTwoNumbers(l1,l2))

class ReverseLinkedList:

    #迭代法反转列表
    #一次从旧列表中取结点插入新列表的头部
    def reverseList(self,head):
        '''
        :type head:ListNode
        :return:
        '''
        newHead=None   #记录新列表的第一个结点
        curNode=head   #旧列表中的当前结点
        while curNode: #迭代至链尾
            tmp=curNode.next   #保存旧结点中下个节点
            curNode.next=newHead  #反转
            newHead=curNode   #移动头结点
            curNode=tmp   #移动当前结点
        return newHead



    #递归实现列表反转
    def recursiveReverse(self,head):
        '''
        :type head:ListNode
        :return:
        '''
        if head==None or head.next==None:
            return head
        #假设原始链表为1->2->3->4->None
        #执行完下一步后 :
        # head=1->2->None
        # newHead=4->3->2->None
        newHead=self.reverseList(head.next)
        head.next.next=head
        head.next=None
        return newHead

    def leetcodeReverse(self,head):
        if not head:
            return head
        dummy=ListNode(float('inf')) #添加一个头结点
        dummy.next=head

        tail=dummy.next
        cur=tail.next
        while cur:
            #将cur放到链表头部
            tail.next=cur.next
            cur.next=dummy.next
            dummy.next=cur
            cur=tail.next
        return dummy.next







    def run(self):
        l1 = ListNode(2)
        l1.next = ListNode(4)
        l1.next.next = ListNode(3)
        printLinkedList(l1)
        print()
        #printLinkedList(self.reverseList(l1))
        print()
        #printLinkedList(self.recursiveReverse(l1))
        printLinkedList(self.leetcodeReverse(l1))




class ReverseLinkedList2:

    def reverseBetween(self, head, m, n):
        '''
        :type head:ListNode
        :type m: int
        :type n: int
        :return:
        '''
        ''' 这是我原先的解法，下面有更好的leetcode解法
        mPrev=head if m>1 else None
        start=head
        nNext=head
        for i in range(0,n):
            if i<m-2: #走m-2步到达m的前驱
                mPrev=mPrev.next
            if i<m-1: #走m-1到达m
                start=start.next
            nNext=nNext.next #走n步到达n的后继

        #对[m,n]链表进行反转
        cur=start
        newHead=nNext
        while cur!=nNext:
            tmp=cur.next
            cur.next=newHead
            newHead=cur
            cur=tmp
        if mPrev:
            mPrev.next=newHead
            return head
        else:
            return newHead
        '''
        dummyHead=ListNode(float('inf')) #给链表添加一个头结点
        dummyHead.next=head

        tail=dummyHead
        for i in range(m-1): #tail指向m节点的前驱
            tail=tail.next
        head2=tail   #给反转的子链表添加一个头结点

        tail=head2.next
        cur=tail.next
        for i in range(m,n):
            tail.next=cur.next #将cur从链表中取出，并插入到头部
            cur.next=head2.next
            head2.next=cur
            cur=tail.next
        return dummyHead.next



    def run(self):
        l1=ListNode(1)
        l2=ListNode(2)
        l3=ListNode(3)
        l4=ListNode(4)
        l5=ListNode(5)

        l1.next=l2
        l2.next=l3
        l3.next=l4
        l4.next=l5

        printLinkedList(l1)
        print()
        printLinkedList(self.reverseBetween(l1, 5, 5))

class PartitionList:

    def partition(self,head,x):

        left_dummy=ListNode(float('inf'))
        right_dummy=ListNode(float('inf'))
        left_tail=left_dummy
        right_tail=right_dummy

        cur=head
        while cur:
            if cur.val<x:
                left_tail.next=cur
                left_tail=cur
            else:
                right_tail.next=cur
                right_tail=cur
            cur=cur.next
        left_tail.next=right_dummy.next
        right_tail.next=None #这一步不要忘了
        return left_dummy.next

#2.2.4给定有序列表，删除重复元素
class RemoveDuplicatesFromSortedList:

    def deleteDuplicates(self,head):
        '''
        :type head:ListNode
        :param head:
        :return:
        '''
        if head==None:
            return head
        tail=head #type:ListNode
        cur=head.next
        while cur:
            if cur.val==tail.val:
                tail.next=cur.next
                del cur
            else:
                tail=cur
            cur=tail.next
        return head

    def recursiveRemove(self,head):
        '''
        :type head:ListNode
        :param head:
        :return:
        '''
        if head==None or head.next==None:
            return head
        head2=self.recursiveRemove(head.next)
        if head.val==head2.val:
            head.next=head2.next
        return head


#2.2.5 只保留出现过一次的结点
class RemoveDuplicatesFromSortedList2:

    def recursiveRemove(self,head):
        '''
        :type head:ListNode
        :param head:
        :return:
        '''
        if head==None or head.next==None:
            return head
        cur=head.next
        if cur.val==head.val:
            while cur and head.val==cur.val:
                cur=cur.next
            return self.recursiveRemove(cur)
        else:
            head.next=self.recursiveRemove(cur)
            return head

    def removeDuplicates(self,head):
        '''
        :type head:ListNode
        :param head:
        :return:
        '''
        if head==None or head.next==None:
            return head
        dummy=ListNode(float('inf')) #新链表头结点
        tail=dummy   #新链表尾节点
        cur=head
        while cur!=None:
            duplicated=False
            #如果cur结点元素重复出现了
            while cur.next!=None and cur.next.val==cur.val:
                cur=cur.next
                duplicated = True  #表明当前cur是重复出现过的
            if duplicated:
                cur=cur.next
                continue
            tail.next=cur
            tail=tail.next
            cur=cur.next
        tail.next=None
        return dummy.next

    def run(self):
        head=ListNode(1)
        node2=ListNode(2)
        node3=ListNode(2)
        head.next=node2
        node2.next=node3

        printLinkedList(head)
        print()
        printLinkedList(self.recursiveRemove(head))

#2.2.6 旋转链表
class RotateList:

    def rotate(self,head,k):
        if head==None:
            return head
        cur=head
        n=1
        while cur.next!=None:
            n+=1
            cur=cur.next
        cur.next=head #将链表链接成环
        k%=n
        p=head
        for i in range(n-k-1):
            p=p.next
        head=p.next
        p.next=None
        return head

#2.2.7 删除链表倒数第n个节点
class RemoveNthFromEnd:

    def remove(self,head,n):
        '''
        :type head:ListNode
        :param head:
        :param n:
        :return:
        '''
        dummy=ListNode(-1)
        dummy.next=head
        p,q=dummy,dummy
        for i in range(n):
            q=q.next
        while q.next!=None:  #q走到倒数第一个元素，p走到倒数第n+1个元素
            q=q.next
            p=p.next
        #删除倒数第n个节点
        p.next=p.next.next
        return dummy.next


class  SwapNodesInPairs:

    def recursiveSwap(self, head):
        '''
        :type head:ListNode
        :param head:
        :return:
        '''
        if head==None or head.next==None:
            return head
        p=head.next
        head.next=self.recursiveSwap(p.next)
        p.next=head
        return p

    def swap(self,head):
        '''
        :type head:ListNode
        :param head:
        :return:
        '''
        if head == None or head.next == None:
            return head
        dummy = ListNode(-1)
        dummy.next = head
        tail = dummy  # 转换完部分的尾结点
        p, q = head, None
        while p != None and p.next != None:
            q = p.next
            p.next = q.next   #交换p，q需要改变3次
            q.next = p
            tail.next = q
            tail = p   #不要忘了更新尾结点
            p = p.next
        return dummy.next

class ReverseNodeInKGroup:

    def recursiveReverseKGroup(self, head, k):
        '''
        :type head:ListNode
        :param head:
        :param k:
        :return:
        '''
        if head==None or head.next==None or k==1:
            return head
        #处理base case
        cur=head
        for i in range(k):
            if cur==None: #链表长度小于k
                return head
            cur=cur.next
        #处理递归情况
        dummy=ListNode(-1)
        dummy.next=head
        tail=head
        for i in range(k-1):  #反转k个节点
            tmp=tail.next
            tail.next=tmp.next  #头插法
            tmp.next=dummy.next
            dummy.next=tmp
        tail.next=self.recursiveReverseKGroup(tail.next, k) #处理递归情况
        return dummy.next

    def reverseKGroup(self,head,k):
        if head == None or head.next == None or k == 1:
            return head
        dummy = ListNode(-1)
        dummy.next = head
        head2 = dummy
        while head2.next != None:
            end = head2.next
            enough = True    #head2之后是否有k个结点供反转
            for i in range(k):
                if end == None:
                    enough = False
                    break
                end = end.next
            if enough:  # 够k个，反转
                # 在(head2,end)区间进行反转
                # 反转后head2.next成为该区间最后一个结点，即为下一个head2
                lastHead = cur = head2.next
                rhead = end
                while cur != end:
                    tmp = cur.next
                    cur.next = rhead
                    head2.next = cur
                    rhead = cur
                    cur = tmp
                head2 = lastHead
            else:  # 不够k个跳出循环
                break
        return dummy.next



class CopyListWithRandomPointer:

    def deepCopy(self,head):
        '''
        :type head:RandomListNode
        :param head:
        :return:
        '''
        #复制结点,暂不处理random指针
        cur=head
        while cur!=None:
            node=RandomListNode(cur.label)
            node.next=cur.next
            cur.next=node
            cur=node.next
        #处理random指针
        cur=head
        while cur!=None:
            node=cur.next
            if cur.random!=None:
                node.random=cur.random.next
            cur=node.next
        #拆分成两条链表
        dummy=RandomListNode(-1)
        tail=dummy
        cur=head
        while cur!=None:
            tail.next=cur.next
            tail=tail.next
            cur.next=cur.next.next
            cur=cur.next
        return dummy.next


class LinkedListCycle:

    def hasCycle(self,head):
        '''
        :type head:ListNode
        :param head:
        :return:
        '''
        fast=head
        slow=head
        #由于fast要走两步，所以要检查fast.next
        while fast!=None and fast.next!=None:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:
                return True
        return False

class LinkedListCycle2:

    #返回链表的环入口
    def deleteCycle(self,head):
        fast=head
        slow=head
        while fast!=None and fast.next!=None:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:#存在环,开始找环入口
                slow2=head
                while slow2!=slow:
                    slow=slow.next
                    slow2=slow2.next
                return slow2
        return None #没有环


class ReorderList:

    def reorder(self,head):
        #找到链表的中点，从中点断开成两条链表
        if head==None or head.next==None:
            return head
        tail=None
        slow=head
        fast=head
        while fast!=None and fast.next!=None: #fast要走两步
            tail=slow
            slow=slow.next
            fast=fast.next.next
        tail.next=None
        #将后半部分反转
        head2=self.reverse(slow)
        #合并两条链表
        cur1=head
        cur2=head2
        while cur1.next!=None:
            next2=cur2.next
            cur2.next=cur1.next
            cur1.next=cur2
            cur1=cur2.next
            cur2=next2
        cur1.next=cur2
        return head



    def reverse(self,head):
        if head==None or head.next==None:
            return  head
        newHead=None
        cur=head
        while cur!=None:
            tmp=cur.next
            cur.next=newHead
            newHead=cur
            cur=tmp
        return newHead


def main():
    #sol=ReverseLinkedList()
    #sol=ReverseLinkedList2()
    sol=RemoveDuplicatesFromSortedList2()
    sol.run()

if __name__ == '__main__':
    main()
